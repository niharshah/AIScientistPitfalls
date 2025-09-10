# Remove-Gating-Mechanism ablation – self-contained script
import os, pathlib, random, math, time, numpy as np, torch
from collections import Counter
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

# --------------------------------------------------------------------- #
# experiment logging dict
experiment_data = {
    "Remove_Gating_Mechanism": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
exp_key = "Remove_Gating_Mechanism"

# --------------------------------------------------------------------- #
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------------------------- #
# load SPR-BENCH (or fallback synthetic)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


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
    print("SPR_BENCH not found – using synthetic toy data.")
    spr = DatasetDict(
        {"train": synth_ds(1200), "dev": synth_ds(300), "test": synth_ds(300)}
    )

num_labels = len(set(spr["train"]["label"]))
print("Number of labels:", num_labels)

# --------------------------------------------------------------------- #
# vocabulary over characters
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
print("Vocab size:", vocab_size, "| MAX_LEN:", MAX_LEN)

# --------------------------------------------------------------------- #
# hashed bigram util
BIGRAM_DIM = 64


def bigram_hash(a, b):
    return (hash(a + ":" + b) & 0xFFFFFFFF) % BIGRAM_DIM


# --------------------------------------------------------------------- #
# torch dataset
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab = hf_ds["sequence"], hf_ds["label"]
        self.count_dim = vocab_size - 1  # exclude PAD

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        ids = [vocab.get(c, PAD_ID) for c in s[:MAX_LEN]]
        attn = [1] * len(ids)
        if len(ids) < MAX_LEN:
            pad = [PAD_ID] * (MAX_LEN - len(ids))
            ids.extend(pad)
            attn.extend([0] * len(pad))
        # unigram counts
        cnt = torch.zeros(self.count_dim, dtype=torch.float32)
        for ch, n in Counter(s).items():
            vi = vocab.get(ch, 0)
            if vi > 0:
                cnt[vi - 1] = float(n)
        cnt /= len(s)
        # bigram counts
        big = torch.zeros(BIGRAM_DIM, dtype=torch.float32)
        for a, b in zip(s, s[1:]):
            big[bigram_hash(a, b)] += 1.0
        if len(s) > 1:
            big /= len(s) - 1
        length_feat = torch.tensor([len(s) / MAX_LEN], dtype=torch.float32)
        sym_feat = torch.cat([cnt, big, length_feat], dim=0)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attn),
            "sym_feat": sym_feat,
            "labels": torch.tensor(self.lab[idx]),
        }


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_loader = DataLoader(SPRTorchDataset(spr["train"]), 128, True, collate_fn=collate)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), 256, False, collate_fn=collate)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), 256, False, collate_fn=collate)


# --------------------------------------------------------------------- #
# model – same backbone but WITHOUT gating
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ConcatHybridTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 128
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PosEnc(d_model, MAX_LEN)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 4, 256, 0.1, batch_first=True), 2
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        sym_in = (vocab_size - 1) + BIGRAM_DIM + 1
        self.sym_proj = nn.Sequential(
            nn.Linear(sym_in, 64), nn.ReLU(), nn.LayerNorm(64)
        )
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(d_model + 64, num_labels)

    def forward(self, input_ids, attention_mask, sym_feat):
        x = self.embed(input_ids)  # [B,L,dm]
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B,dm]
        s = self.sym_proj(sym_feat)  # [B,64]
        h = torch.cat([x, s], dim=-1)  # concat, no gating
        h = self.dropout(h)
        return self.cls(h)


model = ConcatHybridTransformer().to(device)


# --------------------------------------------------------------------- #
# criterion, optimiser, scheduler
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_class = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        with torch.no_grad():
            true = torch.zeros_like(log_probs)
            true.fill_(self.smoothing / (n_class - 1))
            true.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true * log_probs, dim=1))


criterion = LabelSmoothingCE(0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


# --------------------------------------------------------------------- #
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["sym_feat"]
            )
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    f1 = f1_score(gts, preds, average="macro")
    return tot_loss / len(loader.dataset), f1, preds, gts


# --------------------------------------------------------------------- #
EPOCHS = 10
start = time.time()

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    sched.step()

    d = experiment_data[exp_key]["SPR_BENCH"]
    d["losses"]["train"].append(tr_loss)
    d["losses"]["val"].append(val_loss)
    d["metrics"]["train"].append(tr_f1)
    d["metrics"]["val"].append(val_f1)
    d["epochs"].append(ep)

    print(f"Epoch {ep}: val_loss={val_loss:.4f}, MacroF1={val_f1:.4f}")

test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, False)
print(f"Test: loss={test_loss:.4f}, MacroF1={test_f1:.4f}")

d = experiment_data[exp_key]["SPR_BENCH"]
d["predictions"], d["ground_truth"] = test_preds, test_gts
d["test_loss"], d["test_macroF1"] = test_loss, test_f1

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved all metrics to", os.path.join(working_dir, "experiment_data.npy"))
print("Total runtime: %.1fs" % (time.time() - start))
