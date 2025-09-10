# -----------------------------------------------------------
# Remove_Label_Smoothing_Loss – full, self-contained ablation
# -----------------------------------------------------------
import os, pathlib, random, math, numpy as np, torch, time
from collections import Counter
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

# -----------------------------------------------------------
# experiment dict
experiment_data = {
    "Remove_Label_Smoothing_Loss": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
exp_rec = experiment_data["Remove_Label_Smoothing_Loss"]["SPR_BENCH"]

# -----------------------------------------------------------
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------
# load SPR_BENCH or synthetic set
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def synth_ds(n_rows=300, max_len=18, n_labels=6):
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWX")
    data = {"id": [], "sequence": [], "label": []}
    for i in range(n_rows):
        seq = "".join(random.choices(alphabet, k=random.randint(5, max_len)))
        data["id"].append(str(i))
        data["sequence"].append(seq)
        data["label"].append(random.randint(0, n_labels - 1))
    return load_dataset("json", data_files={"train": [data]}, split="train")


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if data_root.exists():
    spr = load_spr_bench(data_root)
else:
    print("SPR_BENCH not found – using synthetic data.")
    spr = DatasetDict(
        {
            "train": synth_ds(1200),
            "dev": synth_ds(300),
            "test": synth_ds(300),
        }
    )

num_labels = len(set(spr["train"]["label"]))
print("Number of labels:", num_labels)

# -----------------------------------------------------------
# vocab
PAD_ID = 0


def build_vocab(ds):
    chars = set()
    for s in ds["sequence"]:
        chars.update(s)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)
MAX_LEN = min(128, max(len(s) for s in spr["train"]["sequence"]))
BIGRAM_DIM = 64


def bigram_hash(a, b):
    return (hash(a + ":" + b) & 0xFFFFFFFF) % BIGRAM_DIM


# -----------------------------------------------------------
# torch Dataset
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab = hf_ds["sequence"], hf_ds["label"]
        self.count_dim = vocab_size - 1

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        ids = [vocab.get(c, PAD_ID) for c in s[:MAX_LEN]]
        attn = [1] * len(ids)
        pad_len = MAX_LEN - len(ids)
        if pad_len:
            ids += [PAD_ID] * pad_len
            attn += [0] * pad_len
        cnt = torch.zeros(self.count_dim, dtype=torch.float32)
        for ch, n in Counter(s).items():
            vi = vocab.get(ch, 0)
            if vi:
                cnt[vi - 1] = n
        cnt /= len(s)
        big = torch.zeros(BIGRAM_DIM, dtype=torch.float32)
        for a, b in zip(s, s[1:]):
            big[bigram_hash(a, b)] += 1.0
        if len(s) > 1:
            big /= len(s) - 1
        length_feat = torch.tensor([len(s) / MAX_LEN], dtype=torch.float32)
        sym = torch.cat([cnt, big, length_feat], 0)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attn),
            "sym_feat": sym,
            "labels": torch.tensor(self.lab[idx]),
        }


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_ds, dev_ds, test_ds = map(
    SPRTorchDataset, (spr["train"], spr["dev"], spr["test"])
)
train_loader = DataLoader(train_ds, 128, True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, False, collate_fn=collate)


# -----------------------------------------------------------
# model
class PosEnc(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class GatedHybridTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 128
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PosEnc(d_model, MAX_LEN)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 4, 256, 0.1, batch_first=True), 2
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        sym_in = vocab_size - 1 + BIGRAM_DIM + 1
        self.sym_proj = nn.Sequential(
            nn.Linear(sym_in, 64), nn.ReLU(), nn.LayerNorm(64)
        )
        self.gate_fc = nn.Linear(64, d_model)
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(d_model + 64, num_labels)

    def forward(self, ids, attn, sym_feat):
        x = self.embed(ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=(attn == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        s = self.sym_proj(sym_feat)
        x = x * torch.sigmoid(self.gate_fc(s))
        h = self.dropout(torch.cat([x, s], -1))
        return self.cls(h)


model = GatedHybridTransformer().to(device)

# -----------------------------------------------------------
# loss / optimiser / sched (no label smoothing)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)


# -----------------------------------------------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    tots, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["sym_feat"]
            )
            loss = criterion(logits, batch["labels"])
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            tots += loss.item() * batch["labels"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    return tots / len(loader.dataset), f1_score(gts, preds, average="macro"), preds, gts


# -----------------------------------------------------------
EPOCHS = 10
start = time.time()
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    sched.step()
    for k, v in [("train", tr_loss), ("val", val_loss)]:
        exp_rec["losses"][k].append(v)
    for k, v in [("train", tr_f1), ("val", val_f1)]:
        exp_rec["metrics"][k].append(v)
    exp_rec["epochs"].append(ep)
    print(f"Epoch {ep}: val_loss={val_loss:.4f}, MacroF1={val_f1:.4f}")

test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, False)
exp_rec["predictions"] = test_preds
exp_rec["ground_truth"] = test_gts
exp_rec["test_loss"] = test_loss
exp_rec["test_macroF1"] = test_f1
print(f"Test: loss={test_loss:.4f}, MacroF1={test_f1:.4f}")

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy to", working_dir)
print("Total runtime: %.1fs" % (time.time() - start))
