import os, random, csv, time, pathlib
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# runtime / path setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------------------------------------------------------
# metric helpers (unchanged)
def _count_shape(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def _count_color(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def scaa(seqs, y_true, y_pred):
    w = [_count_shape(s) + _count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ------------------------------------------------------------
# synthetic-data generation for three distinct latent rules
def _gen_dataset(rule, n=3000, max_len=8):
    shapes, colors = list("ABC"), list("123")
    rows = []
    for _ in range(n):
        toks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, max_len))
        ]
        rows.append({"sequence": " ".join(toks), "label": rule(toks)})
    return rows


# rule 1: parity of token 'A1'
rule_a1 = lambda toks: sum(t == "A1" for t in toks) % 2
# rule 2: parity of token 'B2'
rule_b2 = lambda toks: sum(t == "B2" for t in toks) % 2


# rule 3: majority shape == 'A'
def rule_majority_A(toks):
    shapes = [t[0] for t in toks]
    return int(shapes.count("A") > len(shapes) // 2)


DATASETS = {}
for name, rule in [
    ("A1_parity", rule_a1),
    ("B2_parity", rule_b2),
    ("MajorityA", rule_majority_A),
]:
    DATASETS[name] = {
        "train": _gen_dataset(rule, 4000),
        "dev": _gen_dataset(rule, 1000),
        "test": _gen_dataset(rule, 1000),
    }

# ------------------------------------------------------------
# vocabulary over union of all datasets
PAD, CLS = "<PAD>", "<CLS>"
vocab = set([PAD, CLS])
for d in DATASETS.values():
    for split in d.values():
        for r in split:
            vocab.update(r["sequence"].split())
itos = list(vocab)
stoi = {t: i for i, t in enumerate(itos)}


def encode(seq, max_len=20):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()]
    ids = ids[:max_len] + [stoi[PAD]] * (max_len - len(ids))
    return ids


vocab_size = len(itos)


# ------------------------------------------------------------
# contrastive augmentation
def aug(seq: str) -> str:
    toks = seq.split()
    if len(toks) > 1 and random.random() < 0.7:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    toks = [t for t in toks if random.random() > 0.1]
    if not toks:
        toks = ["A1"]
    return " ".join(toks)


# ------------------------------------------------------------
# PyTorch dataset wrappers
class Contrastive(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        return torch.tensor(encode(aug(s), self.max_len)), torch.tensor(
            encode(aug(s), self.max_len)
        )


class Labelled(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(encode(r["sequence"], self.max_len)),
            torch.tensor(r["label"]),
            r["sequence"],
        )


# ------------------------------------------------------------
# model definitions
class Encoder(nn.Module):
    def __init__(self, vocab, dm=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dm, padding_idx=stoi[PAD])
        self.gru = nn.GRU(dm, dm, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * dm, dm)

    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)
        h = torch.cat([h[0], h[1]], 1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, enc, nc):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.proj.out_features, nc)

    def forward(self, x):
        feat = self.enc(x)
        return self.head(feat), feat


def nt_xent(feats, temp=0.5):
    f = F.normalize(feats, dim=1)
    N = f.size(0) // 2
    sim = torch.mm(f, f.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(N, 2 * N, device=f.device)
    targets = torch.cat([targets, torch.arange(0, N, device=f.device)])
    return F.cross_entropy(sim, targets)


# ------------------------------------------------------------
# training / evaluation setup
BATCH = 128
EPOCH_PRE = 2
EPOCH_FT = 2
MAX_LEN = 20
num_classes = 2
enc = Encoder(vocab_size).to(device)
model = Classifier(enc, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# loaders: union training rows
union_train = [r for d in DATASETS.values() for r in d["train"]]
contrast_loader = DataLoader(
    Contrastive(union_train, MAX_LEN), batch_size=BATCH, shuffle=True
)
ft_loader = DataLoader(Labelled(union_train, MAX_LEN), batch_size=BATCH, shuffle=True)
dev_loaders = {
    name: DataLoader(Labelled(ds["dev"], MAX_LEN), batch_size=BATCH)
    for name, ds in DATASETS.items()
}

# storage dict
experiment_data = {
    "multi_synth_rule_diversity": {
        name: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
        for name in DATASETS.keys()
    }
}

# ------------------------------------------------------------
# contrastive pre-training
for ep in range(1, EPOCH_PRE + 1):
    model.train()
    tot = 0
    for v1, v2 in contrast_loader:
        v1 = v1.to(device)
        v2 = v2.to(device)
        _, f1 = model(v1)
        _, f2 = model(v2)
        loss = nt_xent(torch.cat([f1, f2], 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += loss.item() * v1.size(0)
    print(f"Contrastive {ep}: loss {tot/len(union_train):.4f}")

# ------------------------------------------------------------
# fine-tuning
criterion = nn.CrossEntropyLoss()
for ep in range(1, EPOCH_FT + 1):
    model.train()
    tr_loss = 0
    for ids, labels, _ in ft_loader:
        ids, labels = ids.to(device), labels.to(device)
        logits, _ = model(ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * ids.size(0)
    tr_loss /= len(union_train)

    # per-dataset validation
    model.eval()
    with torch.no_grad():
        for name, loader in dev_loaders.items():
            val_loss = 0
            preds = []
            gts = []
            seqs = []
            for ids, labels, seq in loader:
                ids, labels = ids.to(device), labels.to(device)
                logits, _ = model(ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * ids.size(0)
                preds.extend(torch.argmax(logits, 1).cpu().tolist())
                gts.extend(labels.cpu().tolist())
                seqs.extend(seq)
            val_loss /= len(DATASETS[name]["dev"])
            SWA = shape_weighted_accuracy(seqs, gts, preds)
            CWA = color_weighted_accuracy(seqs, gts, preds)
            SCAA = scaa(seqs, gts, preds)
            print(
                f"Epoch {ep} | {name}: loss={val_loss:.4f} SWA={SWA:.3f} CWA={CWA:.3f} SCAA={SCAA:.3f}"
            )
            # logging
            ex = experiment_data["multi_synth_rule_diversity"][name]
            ex["metrics"]["train"].append({"SWA": None, "CWA": None, "SCAA": None})
            ex["metrics"]["val"].append({"SWA": SWA, "CWA": CWA, "SCAA": SCAA})
            ex["losses"]["train"].append(tr_loss)
            ex["losses"]["val"].append(val_loss)
            ex["predictions"] = preds
            ex["ground_truth"] = gts
            ex["epochs"].append(ep)

# ------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved to", os.path.join(working_dir, "experiment_data.npy"))
