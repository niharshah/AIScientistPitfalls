import os, random, pathlib, numpy as np, torch, itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------- mandatory dirs / devices -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- utility metrics --------------------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split()})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def harmonic_mean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ---------------- data loading (real or synthetic) ---------------
def load_real_spr(root):
    from datasets import load_dataset

    def _load(name):
        return list(
            load_dataset(
                "csv",
                data_files=str(root / name),
                split="train",
                cache_dir=".cache_dsets",
            )
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


def create_synth(n):
    shapes, colors = "ABCD", "1234"

    def make_seq():
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(8))

    def label(seq):
        return max(
            set([t[0] for t in seq.split()]), key=[t[0] for t in seq.split()].count
        )

    return [
        {"id": i, "sequence": (s := make_seq()), "label": label(s)} for i in range(n)
    ]


DATA_DIR = pathlib.Path("SPR_BENCH")
try:
    data = load_real_spr(DATA_DIR)
except:
    print("Real SPR_BENCH not found, generating synthetic data")
    data = {
        "train": create_synth(4000),
        "dev": create_synth(1000),
        "test": create_synth(1000),
    }

# ---------------- glyph vector & clustering ----------------------
all_tokens = [
    tok
    for row in itertools.chain(data["train"], data["dev"], data["test"])
    for tok in row["sequence"].split()
]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2i = {s: i for i, s in enumerate(shapes)}
color2i = {c: i for i, c in enumerate(colors)}


def tok_vec(t):
    return np.array([shape2i[t[0]], color2i[t[1]]], dtype=np.float32)


token_vecs = np.stack([tok_vec(t) for t in all_tokens])
# pick best k via silhouette on sample
candidate_k = [6, 8, 10, 12, 14]
sil_scores = []
sample_idx = np.random.choice(
    len(token_vecs), size=min(3000, len(token_vecs)), replace=False
)
for k in candidate_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(token_vecs[sample_idx])
    sil_scores.append(silhouette_score(token_vecs[sample_idx], km.labels_))
best_k = candidate_k[int(np.argmax(sil_scores))]
print(f"Chosen number of clusters: {best_k} (silhouette={max(sil_scores):.3f})")

kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=1).fit(token_vecs)
train_clusters_present = set(
    kmeans.predict(
        np.stack(
            [
                tok_vec(t)
                for t in [tok for r in data["train"] for tok in r["sequence"].split()]
            ]
        )
    )
)


# -------------- feature extraction per sequence ------------------
def sequence_features(seq: str):
    toks = seq.split()
    clust = kmeans.predict(np.stack([tok_vec(t) for t in toks]))
    hist = np.bincount(clust, minlength=best_k) / len(toks)
    shape_hist = np.bincount(
        [shape2i[t[0]] for t in toks], minlength=len(shapes)
    ) / len(toks)
    color_hist = np.bincount(
        [color2i[t[1]] for t in toks], minlength=len(colors)
    ) / len(toks)
    extra = np.array(
        [count_shape_variety(seq), count_color_variety(seq), clust.mean() / best_k],
        dtype=np.float32,
    )
    return np.concatenate([hist, shape_hist, color_hist, extra])


feat_dim = len(sequence_features(data["train"][0]["sequence"]))


# ---------------- PyTorch dataset --------------------------------
class SPRSet(Dataset):
    def __init__(self, rows):
        self.seq = [r["sequence"] for r in rows]
        self.x = np.stack([sequence_features(s) for s in self.seq]).astype(np.float32)
        self.y = le.transform([r["label"] for r in rows]).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx]),
            "seq": self.seq[idx],
        }


le = LabelEncoder()
le.fit([r["label"] for r in data["train"]])
train_ds, dev_ds, test_ds = (
    SPRSet(data["train"]),
    SPRSet(data["dev"]),
    SPRSet(data["test"]),
)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=256)
test_dl = DataLoader(test_ds, batch_size=256)

# ---------------- model ------------------------------------------
model = nn.Sequential(
    nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, len(le.classes_))
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# ------------ experiment_data skeleton ---------------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------- helpers for OCGA -------------------------------
def ocga(seqs, y_t, y_p):
    acc_cnt = tot = 0
    for s, t, p in zip(seqs, y_t, y_p):
        clusts = set(kmeans.predict(np.stack([tok_vec(tok) for tok in s.split()])))
        if not clusts.issubset(train_clusters_present):  # OOC
            tot += 1
            if t == p:
                acc_cnt += 1
    return acc_cnt / max(1, tot)


# ---------------- training loop with early stopping --------------
patience, wait, best_cshm = 8, 0, -1
best_state, best_epoch = None, 0
for epoch in range(1, 51):
    # train
    model.train()
    tr_loss = 0
    for batch in train_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt.zero_grad()
        out = model(batch["x"])
        loss = crit(out, batch["y"])
        loss.backward()
        opt.step()
        tr_loss += loss.item() * batch["y"].size(0)
    tr_loss /= len(train_ds)
    experiment_data["SPR"]["losses"]["train"].append((epoch, tr_loss))

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
            out = model(batch["x"])
            val_loss += crit(out, batch["y"]).item() * batch["y"].size(0)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            g = batch["y"].cpu().numpy()
            gts.extend(g)
            seqs.extend(batch["seq"])
    val_loss /= len(dev_ds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hm = harmonic_mean(cwa, swa)
    oca = ocga(seqs, gts, preds)
    experiment_data["SPR"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR"]["metrics"]["val"].append((epoch, cwa, swa, hm, oca))
    print(
        f"Epoch {epoch:02d}: val_loss={val_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} HM={hm:.3f} OCGA={oca:.3f}"
    )
    # early stop
    if hm > best_cshm + 1e-4:
        best_cshm, best_state, best_epoch, wait = (
            hm,
            {k: v.cpu() for k, v in model.state_dict().items()},
            epoch,
            0,
        )
    else:
        wait += 1
    if wait >= patience:
        print(f"Early stopping at epoch {epoch} (best={best_epoch})")
        break

# ---------------- test evaluation --------------------------------
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
        out = model(batch["x"])
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["y"].cpu().numpy())
        seqs.extend(batch["seq"])
experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
print(
    "TEST  CWA={:.3f}  SWA={:.3f}  OCGA={:.3f}".format(
        color_weighted_accuracy(seqs, gts, preds),
        shape_weighted_accuracy(seqs, gts, preds),
        ocga(seqs, gts, preds),
    )
)

# ---------------- save -------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
