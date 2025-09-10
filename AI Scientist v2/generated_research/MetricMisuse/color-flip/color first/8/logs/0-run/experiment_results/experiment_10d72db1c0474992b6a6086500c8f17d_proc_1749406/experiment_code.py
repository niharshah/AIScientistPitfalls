# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, pathlib, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# ---------- mandatory dirs / device ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics -------------------------------------------------
def count_color_variety(seq):
    return len({t[1] for t in seq.split()})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split()})


def CWA(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def SWA(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def hmean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ---------- load real or synthetic SPR ------------------------------
def load_real(root):
    from datasets import load_dataset

    def _ld(f):
        return list(load_dataset("csv", data_files=str(root / f), split="train"))

    return {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}


def synth(n):
    shapes, colors = "ABCD", "1234"
    mk = lambda: " ".join(
        random.choice(shapes) + random.choice(colors) for _ in range(8)
    )
    lab = lambda s: max([t[0] for t in s.split()], key=[t[0] for t in s.split()].count)
    return [{"id": i, "sequence": (s := mk()), "label": lab(s)} for i in range(n)]


DATA_DIR = pathlib.Path("SPR_BENCH")
try:
    data = load_real(DATA_DIR)
except Exception:
    print("Real SPR_BENCH not found, using synthetic")
    data = {"train": synth(4000), "dev": synth(1000), "test": synth(1000)}

# ---------- glyph → vector (shape only) & clustering ----------------
all_toks = [
    tok for row in itertools.chain(*data.values()) for tok in row["sequence"].split()
]
shapes = sorted({t[0] for t in all_toks})
colors = sorted({t[1] for t in all_toks})
s2i = {s: i for i, s in enumerate(shapes)}
c2i = {c: i for i, c in enumerate(colors)}


def tokvec(t):  # shape-only vector (1-D)
    return np.array([s2i[t[0]]], np.float32)


vecs = np.stack([tokvec(t) for t in all_toks])
candidate_ks = [2, 3, 4, 5, 6, 8, 10]  # limited reasonable range
sample_idx = np.random.choice(len(vecs), min(3000, len(vecs)), replace=False)
scores = [
    silhouette_score(
        vecs[sample_idx],
        KMeans(k, n_init=8, random_state=0).fit(vecs[sample_idx]).labels_,
    )
    for k in candidate_ks
]
k_best = candidate_ks[int(np.argmax(scores))]
print(f"Chosen k={k_best} (shape only)")
kmeans = KMeans(n_clusters=k_best, n_init=20, random_state=1).fit(vecs)

train_clusters = set(
    kmeans.predict(
        np.stack([tokvec(t) for r in data["train"] for t in r["sequence"].split()])
    )
)

# ---------- sequence to cluster-id list -----------------------------
PAD = 0


def seq2clust(seq):
    ids = (
        kmeans.predict(np.stack([tokvec(t) for t in seq.split()])) + 1
    )  # +1 reserve PAD
    return ids.astype(np.int64)


# ---------- Dataset & DataLoader ------------------------------------
class SPRDataset(Dataset):
    def __init__(self, rows):
        self.seqs = [r["sequence"] for r in rows]
        self.x = [seq2clust(s) for s in self.seqs]
        self.shp = [count_shape_variety(s) for s in self.seqs]
        self.col = [count_color_variety(s) for s in self.seqs]
        self.y = le.transform([r["label"] for r in rows]).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "ids": torch.tensor(self.x[idx]),
            "shc": torch.tensor([self.shp[idx], self.col[idx]], dtype=torch.float32),
            "y": torch.tensor(self.y[idx]),
            "seq": self.seqs[idx],
        }


def collate(batch):
    lens = [len(b["ids"]) for b in batch]
    maxlen = max(lens)
    ids = torch.full((len(batch), maxlen), PAD, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : lens[i]] = b["ids"]
    shc = torch.stack([b["shc"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    seq = [b["seq"] for b in batch]
    return {"ids": ids, "lens": torch.tensor(lens), "shc": shc, "y": y, "seq": seq}


le = LabelEncoder()
le.fit([r["label"] for r in data["train"]])
train_dl = DataLoader(
    SPRDataset(data["train"]), batch_size=256, shuffle=True, collate_fn=collate
)
dev_dl = DataLoader(SPRDataset(data["dev"]), batch_size=512, collate_fn=collate)
test_dl = DataLoader(SPRDataset(data["test"]), batch_size=512, collate_fn=collate)


# ---------- Model ---------------------------------------------------
class GRUReasoner(nn.Module):
    def __init__(self, vocab, embed=32, hid=64, num_cls=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, padding_idx=PAD)
        self.gru = nn.GRU(embed, hid, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hid * 2 + 2, 64), nn.ReLU(), nn.Linear(64, num_cls)
        )

    def forward(self, ids, lens, shc):
        e = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.head(torch.cat([h, shc], dim=1))


model = GRUReasoner(k_best + 1, num_cls=len(le.classes_)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
crit = nn.CrossEntropyLoss()


# ---------- OCGA ----------------------------------------------------
def OCGA(seqs, y_t, y_p):
    tot = acc = 0
    for s, t, p in zip(seqs, y_t, y_p):
        cl = set(kmeans.predict(np.stack([tokvec(tok) for tok in s.split()])))
        if not cl.issubset(train_clusters):
            tot += 1
            acc += int(t == p)
    return acc / max(1, tot)


# ---------- experiment data dict ------------------------------------
experiment_data = {
    "ShapeOnly-KMeans-Clustering": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}
exd = experiment_data["ShapeOnly-KMeans-Clustering"]["SPR"]

# ---------- training ------------------------------------------------
best_hm, best_state, wait = -1, None, 0
for epoch in range(1, 61):
    # train
    model.train()
    tr_loss = 0
    for batch in train_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt.zero_grad()
        out = model(batch["ids"], batch["lens"], batch["shc"])
        loss = crit(out, batch["y"])
        loss.backward()
        opt.step()
        tr_loss += loss.item() * batch["y"].size(0)
    tr_loss /= len(train_dl.dataset)
    exd["losses"]["train"].append((epoch, tr_loss))
    # val
    model.eval()
    val_loss = 0
    preds = []
    gts = []
    seqs = []
    with torch.no_grad():
        for batch in dev_dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["ids"], batch["lens"], batch["shc"])
            val_loss += crit(out, batch["y"]).item() * batch["y"].size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(batch["y"].cpu().numpy())
            seqs.extend(batch["seq"])
    val_loss /= len(dev_dl.dataset)
    cwa, swa = CWA(seqs, gts, preds), SWA(seqs, gts, preds)
    hm, hoc = hmean(cwa, swa), OCGA(seqs, gts, preds)
    exd["losses"]["val"].append((epoch, val_loss))
    exd["metrics"]["val"].append((epoch, cwa, swa, hm, hoc))
    print(
        f"Epoch {epoch:02d}: val_loss={val_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} HM={hm:.3f} OCGA={hoc:.3f}"
    )
    if hm > best_hm + 1e-4:
        best_hm, best_state, wait = (
            hm,
            {k: v.cpu() for k, v in model.state_dict().items()},
            0,
        )
    else:
        wait += 1
    if wait >= 10:
        print("Early stopping.")
        break

# ---------- test ----------------------------------------------------
model.load_state_dict(best_state)
model.to(device)
model.eval()
preds, gts, seqs = [], [], []
with torch.no_grad():
    for batch in test_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["ids"], batch["lens"], batch["shc"])
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["y"].cpu().numpy())
        seqs.extend(batch["seq"])
exd["predictions"], exd["ground_truth"] = preds, gts
print(
    "TEST CWA={:.3f} SWA={:.3f} OCGA={:.3f}".format(
        CWA(seqs, gts, preds), SWA(seqs, gts, preds), OCGA(seqs, gts, preds)
    )
)

# ---------- save ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
