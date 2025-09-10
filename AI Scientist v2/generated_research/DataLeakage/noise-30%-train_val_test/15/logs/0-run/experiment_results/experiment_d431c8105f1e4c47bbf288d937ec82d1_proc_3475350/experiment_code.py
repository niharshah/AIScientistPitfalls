# ------------------------------------------------------------
# Remove_Transformer_Encoder ablation study – single-file script
# ------------------------------------------------------------
import os, pathlib, random, math, time, numpy as np, torch
from collections import Counter
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------
# load SPR-BENCH or create synthetic toy data
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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
    print("SPR_BENCH not found – generating synthetic toy dataset.")
    spr = DatasetDict(
        {"train": synth_ds(1200), "dev": synth_ds(300), "test": synth_ds(300)}
    )

num_labels = len(set(spr["train"]["label"]))
print("Labels:", num_labels)

# ---------------------------------------------------------------------
# build vocab
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
id2char = {i: c for c, i in vocab.items()}
print("Vocab size:", vocab_size)

MAX_LEN = min(128, max(len(s) for s in spr["train"]["sequence"]))

# ---------------------------------------------------------------------
# hashed bigram helper
BIGRAM_DIM = 64


def bigram_hash(a, b):
    return (hash(a + ":" + b) & 0xFFFFFFFF) % BIGRAM_DIM


# ---------------------------------------------------------------------
# torch Dataset
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"]
        self.count_dim = vocab_size - 1

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        ids = [vocab.get(c, PAD_ID) for c in s[:MAX_LEN]]
        attn = [1] * len(ids)
        if len(ids) < MAX_LEN:
            pad = MAX_LEN - len(ids)
            ids += [PAD_ID] * pad
            attn += [0] * pad
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
        sym_feat = torch.cat([cnt, big, length_feat], 0)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attn),
            "sym_feat": sym_feat,
            "labels": torch.tensor(self.lab[idx]),
        }


def collate(b):
    return {k: torch.stack([d[k] for d in b]) for k in b[0]}


train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------------------------------------------------------------------
# Ablated model – NO TransformerEncoder
class GatedBagEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 128
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pool = nn.AdaptiveAvgPool1d(1)  # average over tokens
        sym_in = vocab_size - 1 + BIGRAM_DIM + 1
        self.sym_proj = nn.Sequential(
            nn.Linear(sym_in, 64), nn.ReLU(), nn.LayerNorm(64)
        )
        self.gate_fc = nn.Linear(64, d_model)
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(d_model + 64, num_labels)

    def forward(self, input_ids, attention_mask, sym_feat):
        x = self.embed(input_ids)  # B,L,D
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # B,D
        s = self.sym_proj(sym_feat)  # B,64
        gate = torch.sigmoid(self.gate_fc(s))  # B,D
        x = x * gate  # gated bag-embeddings
        h = torch.cat([x, s], -1)
        h = self.dropout(h)
        return self.cls(h)


model = GatedBagEmbeddings().to(device)


# ---------------------------------------------------------------------
# loss, opt
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.s = smoothing

    def forward(self, logits, target):
        n = logits.size(1)
        logp = torch.log_softmax(logits, 1)
        with torch.no_grad():
            true = torch.zeros_like(logp).fill_(self.s / (n - 1))
            true.scatter_(1, target.unsqueeze(1), 1 - self.s)
        return torch.mean(torch.sum(-true * logp, 1))


criterion = LabelSmoothingCE(0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ---------------------------------------------------------------------
# experiment data dict
experiment_data = {
    "Remove_Transformer_Encoder": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}


# ---------------------------------------------------------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total, preds, gts = 0.0, [], []
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
            total += loss.item() * batch["labels"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    return (
        total / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------------------------------------------------------------------
EPOCHS = 10
start = time.time()
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    sched.step()
    ed = experiment_data["Remove_Transformer_Encoder"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append(tr_f1)
    ed["metrics"]["val"].append(val_f1)
    ed["epochs"].append(epoch)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f} | MacroF1={val_f1:.4f}")

test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, False)
print(f"Test: loss={test_loss:.4f} | MacroF1={test_f1:.4f}")
ed = experiment_data["Remove_Transformer_Encoder"]["SPR_BENCH"]
ed["predictions"] = test_preds
ed["ground_truth"] = test_gts
ed["test_loss"] = test_loss
ed["test_macroF1"] = test_f1

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to:", os.path.join(working_dir, "experiment_data.npy"))
print("Total runtime: %.1fs" % (time.time() - start))
