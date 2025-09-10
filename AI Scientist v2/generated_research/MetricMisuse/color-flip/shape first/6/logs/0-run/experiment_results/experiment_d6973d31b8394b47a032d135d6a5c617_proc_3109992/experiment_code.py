import os, random, pathlib, csv
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------ #
# 1) housekeeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------------------------ #
# 2) metric helpers
def _count_shape(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def _count_color(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def _w_acc(w, y_t, y_p):
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    return _w_acc([_count_shape(s) for s in seqs], y_t, y_p)


def color_weighted_accuracy(seqs, y_t, y_p):
    return _w_acc([_count_color(s) for s in seqs], y_t, y_p)


def scaa(seqs, y_t, y_p):
    w = [_count_shape(s) + _count_color(s) for s in seqs]
    return _w_acc(w, y_t, y_p)


# ------------------------------------------------------------------ #
# 3) dataset (real or synthetic)
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def _load_csv(fp):
    with open(fp) as f:
        rdr = csv.DictReader(f)
        return [{"sequence": r["sequence"], "label": int(r["label"])} for r in rdr]


def _gen_synth(n=3000, mx=8):
    shapes, colors = list("ABC"), list("123")

    def rule(seq):
        return sum(t == "A1" for t in seq) % 2

    rows = []
    for _ in range(n):
        toks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, mx))
        ]
        rows.append({"sequence": " ".join(toks), "label": rule(toks)})
    return rows


dataset: Dict[str, List[Dict]] = {}
try:
    if SPR_PATH.exists():
        for spl in ["train", "dev", "test"]:
            dataset[spl] = _load_csv(SPR_PATH / f"{spl}.csv")
    else:
        raise FileNotFoundError
except Exception:
    print("SPR_BENCH not found – using synthetic data")
    dataset["train"] = _gen_synth(4000)
    dataset["dev"] = _gen_synth(1000)
    dataset["test"] = _gen_synth(1000)
print({k: len(v) for k, v in dataset.items()})

# ------------------------------------------------------------------ #
# 4) vocab / encoding
PAD, CLS = "<PAD>", "<CLS>"
vocab = {PAD, CLS}
for split in dataset.values():
    for r in split:
        vocab.update(r["sequence"].split())
itos = list(vocab)
stoi = {t: i for i, t in enumerate(itos)}


def encode(seq, max_len=20):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()]
    ids = ids[:max_len] + [stoi[PAD]] * (max_len - len(ids))
    return ids


vocab_size = len(itos)


# ------------------------------------------------------------------ #
# 5) augmentations
def aug(seq: str) -> str:
    toks = seq.split()
    if len(toks) > 1 and random.random() < 0.7:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    toks = [t for t in toks if random.random() > 0.1]
    if not toks:
        toks = ["A1"]
    return " ".join(toks)


# ------------------------------------------------------------------ #
# 6) PyTorch datasets
class ContrastiveSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.mx = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        return torch.tensor(encode(aug(s), self.mx)), torch.tensor(
            encode(aug(s), self.mx)
        )


class LabelledSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.mx = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(encode(r["sequence"], self.mx)),
            torch.tensor(r["label"]),
            r["sequence"],
        )


# ------------------------------------------------------------------ #
# 7) model
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.rnn = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        e = self.emb(x)
        _, h = self.rnn(e)
        h = torch.cat([h[0], h[1]], 1)
        return self.proj(h)


class SPRClassifier(nn.Module):
    def __init__(self, enc, n_cls):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.proj.out_features, n_cls)

    def forward(self, x):
        f = self.enc(x)
        return self.head(f), f


def nt_xent(feats, temp=0.5):
    f = F.normalize(feats, dim=1)
    N = f.size(0) // 2
    sim = torch.mm(f, f.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(N, 2 * N, device=f.device)
    targets = torch.cat([targets, torch.arange(0, N, device=f.device)])
    return F.cross_entropy(sim, targets)


# ------------------------------------------------------------------ #
# 8) experiment set-up
BATCH, EPOCH_PRE, EPOCH_FT, MAX_LEN = 128, 2, 2, 20
num_classes = len({r["label"] for r in dataset["train"]})
enc = Encoder(vocab_size).to(device)
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
    "frozen_encoder": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ------------------------------------------------------------------ #
# 9) contrastive pre-training (encoder + head both trainable)
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
    print(f"Contrastive epoch {ep} loss={tot/len(dataset['train']):.4f}")

# ------------------------------------------------------------------ #
# 10) freeze encoder, new optimizer for head only
for p in model.enc.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for ep in range(1, EPOCH_FT + 1):
    # training (head only)
    model.train()
    tr_loss = 0
    for ids, labels, _ in train_loader:
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
        for ids, labels, seq in dev_loader:
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
        f"FT Epoch {ep}: val_loss={val_loss:.4f} | SWA={SWA:.3f} CWA={CWA:.3f} SCAA={SCAA:.3f}"
    )

    # log
    ed = experiment_data["frozen_encoder"]["SPR"]
    ed["metrics"]["train"].append({"SWA": None, "CWA": None, "SCAA": None})
    ed["metrics"]["val"].append({"SWA": SWA, "CWA": CWA, "SCAA": SCAA})
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["epochs"].append(ep)
    ed["predictions"] = preds
    ed["ground_truth"] = gts

# ------------------------------------------------------------------ #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved to", os.path.join(working_dir, "experiment_data.npy"))
