# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, pathlib, csv, time
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# mandatory working dir + device setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# helper functions for metrics
def _count_shape(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def _count_color(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_shape(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_color(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def scaa(seqs, y_true, y_pred):
    w = [_count_shape(s) + _count_color(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


# ---------------------------------------------------------------------
# dataset loading (real or synthetic)
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def _load_csv(fp):
    rows = []
    with open(fp) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({"sequence": r["sequence"], "label": int(r["label"])})
    return rows


def _generate_synth(n=3000, max_len=8):
    shapes, colors = list("ABC"), list("123")

    def rule(seq):
        # trivial parity rule on number of 'A1'
        return sum(tok == "A1" for tok in seq) % 2

    rows = []
    for _ in range(n):
        toks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, max_len))
        ]
        rows.append({"sequence": " ".join(toks), "label": rule(toks)})
    return rows


dataset: Dict[str, List[Dict]] = {}
try:
    if SPR_PATH.exists():
        for split in ["train", "dev", "test"]:
            dataset[split] = _load_csv(SPR_PATH / f"{split}.csv")
    else:
        raise FileNotFoundError
except Exception:
    print("Real SPR_BENCH not found – using synthetic data")
    dataset["train"] = _generate_synth(4000)
    dataset["dev"] = _generate_synth(1000)
    dataset["test"] = _generate_synth(1000)

print({k: len(v) for k, v in dataset.items()})

# ---------------------------------------------------------------------
# vocabulary
PAD, CLS = "<PAD>", "<CLS>"
vocab = {PAD, CLS}
for split in dataset.values():
    for row in split:
        vocab.update(row["sequence"].split())
itos = list(vocab)
stoi = {tok: i for i, tok in enumerate(itos)}
vocab_size = len(itos)


def encode(seq, max_len=20):
    ids = [stoi[CLS]] + [stoi[tok] for tok in seq.split()]
    ids = ids[:max_len] + [stoi[PAD]] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------------
# augmentations for contrastive learning
def aug(seq: str) -> str:
    toks = seq.split()
    if len(toks) > 1 and random.random() < 0.7:  # random swap
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    # token dropout / masking
    toks = [t for t in toks if random.random() > 0.1]
    if not toks:
        toks = ["A1"]  # fallback
    return " ".join(toks)


# ---------------------------------------------------------------------
# PyTorch datasets
class ContrastiveSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        return (
            torch.tensor(encode(aug(s), self.max_len)),
            torch.tensor(encode(aug(s), self.max_len)),
        )


class LabelledSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(encode(r["sequence"], self.max_len)),
            torch.tensor(r["label"]),
            r["sequence"],
        )


# ---------------------------------------------------------------------
# model
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.bigru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)  # concat directions
        return self.proj(h)  # [B,d_model]


class SPRClassifier(nn.Module):
    def __init__(self, enc, n_classes):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.proj.out_features, n_classes)

    def forward(self, x):
        feat = self.enc(x)
        return self.head(feat), feat


# ---------------------------------------------------------------------
def nt_xent(feats, temp=0.5):
    f = F.normalize(feats, dim=1)
    N = f.size(0) // 2
    sim = torch.mm(f, f.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(N, 2 * N, device=f.device)
    targets = torch.cat([targets, torch.arange(0, N, device=f.device)])
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------
# experiment config
BATCH = 128
EPOCH_PRE = 2
EPOCH_FT = 2
MAX_LEN = 20
num_classes = len({r["label"] for r in dataset["train"]})

enc = Encoder(vocab_size, 256).to(device)
model = SPRClassifier(enc, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

contrast_loader = DataLoader(
    ContrastiveSPR(dataset["train"], MAX_LEN), batch_size=BATCH, shuffle=True
)
train_loader = DataLoader(
    LabelledSPR(dataset["train"], MAX_LEN), batch_size=BATCH, shuffle=True
)
dev_loader = DataLoader(LabelledSPR(dataset["dev"], MAX_LEN), batch_size=BATCH)

experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------------------------------------------------------------------
# contrastive pre-training
for ep in range(1, EPOCH_PRE + 1):
    model.train()
    tot = 0
    for v1, v2 in contrast_loader:
        v1, v2 = v1.to(device), v2.to(device)
        _, f1 = model(v1)
        _, f2 = model(v2)
        loss = nt_xent(torch.cat([f1, f2], 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += loss.item() * v1.size(0)
    print(f'Contrastive epoch {ep}: loss = {tot/len(dataset["train"]):.4f}')

# ---------------------------------------------------------------------
# fine-tuning
criterion = nn.CrossEntropyLoss()
for ep in range(1, EPOCH_FT + 1):
    model.train()
    tr_loss = 0
    for batch in train_loader:
        ids, labels, _ = batch
        ids, labels = ids.to(device), labels.to(device)
        logits, _ = model(ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * ids.size(0)
    tr_loss /= len(dataset["train"])

    # validation
    model.eval()
    val_loss = 0
    preds = []
    gts = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            ids, labels, seq = batch
            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)
            loss = criterion(logits, labels)
            val_loss += loss.item() * ids.size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(labels.cpu().tolist())
            seqs.extend(seq)
    val_loss /= len(dataset["dev"])
    SWA = shape_weighted_accuracy(seqs, gts, preds)
    CWA = color_weighted_accuracy(seqs, gts, preds)
    SCAA = scaa(seqs, gts, preds)

    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={SWA:.3f} CWA={CWA:.3f} SCAA={SCAA:.3f}"
    )

    # store
    experiment_data["SPR"]["metrics"]["train"].append(
        {"SWA": None, "CWA": None, "SCAA": None}
    )
    experiment_data["SPR"]["metrics"]["val"].append(
        {"SWA": SWA, "CWA": CWA, "SCAA": SCAA}
    )
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["epochs"].append(ep)
    experiment_data["SPR"]["predictions"] = preds
    experiment_data["SPR"]["ground_truth"] = gts

# ---------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
