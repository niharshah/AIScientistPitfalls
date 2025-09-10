import os, pathlib, random, itertools, time, numpy as np
from collections import Counter
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------- housekeeping -----------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"hidden_dim": {}}  # <-- main container


# ----------------------------- data utils -------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) or 1)


def harmonic_mean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ----------------------------- load / synth data ------------------
def load_real_spr(root: pathlib.Path):
    from datasets import load_dataset

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    splits = {}
    for s in ("train.csv", "dev.csv", "test.csv"):
        splits[s.split(".")[0]] = _load(s)
    return {k: list(v) for k, v in splits.items()}


def create_synthetic_spr(n_train=400, n_dev=100, n_test=100, seq_len=8):
    shapes, colors = list("ABCD"), list("1234")

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
        )

    def make_label(seq):
        return Counter([t[0] for t in seq.split()]).most_common(1)[0][0]

    def make_split(n):
        return [
            {"id": i, "sequence": (seq := make_seq()), "label": make_label(seq)}
            for i in range(n)
        ]

    return {
        "train": make_split(n_train),
        "dev": make_split(n_dev),
        "test": make_split(n_test),
    }


try:
    DATA_PATH = pathlib.Path("SPR_BENCH")
    spr_data = load_real_spr(DATA_PATH)
except Exception:
    print("Real SPR_BENCH not found – using synthetic data.")
    spr_data = create_synthetic_spr()
print({k: len(v) for k, v in spr_data.items()})

# ----------------------------- tokenization & clustering ----------
all_tokens = [tok for r in spr_data["train"] for tok in r["sequence"].split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}


def token_vector(tok: str):
    return [shape2idx[tok[0]], color2idx[tok[1]]]


token_vecs = np.array([token_vector(t) for t in all_tokens])
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(token_vecs)


def sequence_to_features(seq: str) -> np.ndarray:
    tokens = seq.split()
    clusters = kmeans.predict(np.array([token_vector(t) for t in tokens]))
    hist = np.bincount(clusters, minlength=n_clusters) / len(tokens)
    return np.concatenate([hist, [count_shape_variety(seq), count_color_variety(seq)]])


# ----------------------------- dataset ----------------------------
le = LabelEncoder()
le.fit([r["label"] for r in spr_data["train"]])
n_classes = len(le.classes_)


class SPRDataset(Dataset):
    def __init__(self, rows):
        self.seqs = [r["sequence"] for r in rows]
        self.x = np.stack([sequence_to_features(s) for s in self.seqs]).astype(
            np.float32
        )
        self.y = le.transform([r["label"] for r in rows]).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx]),
            "seq": self.seqs[idx],
        }


train_ds, dev_ds, test_ds = (
    SPRDataset(spr_data["train"]),
    SPRDataset(spr_data["dev"]),
    SPRDataset(spr_data["test"]),
)
batch_size = 64
train_loader_base = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# ----------------------------- training loop ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

hidden_dims = [16, 32, 64, 128]
epochs = 15
input_dim = train_ds.x.shape[1]

for hdim in hidden_dims:
    print(f"\n--- training with hidden_dim = {hdim} ---")
    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = nn.Sequential(
        nn.Linear(input_dim, hdim), nn.ReLU(), nn.Linear(hdim, n_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # recreate train loader with shuffle each configuration
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
        train_loss = running_loss / len(train_ds)
        exp_rec["losses"]["train"].append((epoch, train_loss))

        # validation
        model.eval()
        val_loss = 0.0
        preds, gts, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["x"])
                loss = criterion(logits, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                p = logits.argmax(1).cpu().numpy()
                preds.extend(p)
                gts.extend(batch["y"].cpu().numpy())
                seqs.extend(batch["seq"])
        val_loss /= len(dev_ds)
        exp_rec["losses"]["val"].append((epoch, val_loss))
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cshm = harmonic_mean(cwa, swa)
        exp_rec["metrics"]["val"].append((epoch, cwa, swa, cshm))
        print(
            f"Epoch {epoch}: val_loss {val_loss:.4f} | CWA {cwa:.3f} SWA {swa:.3f} CSHM {cshm:.3f}"
        )

    # test predictions
    model.eval()
    test_preds, test_gts, test_seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            p = logits.argmax(1).cpu().numpy()
            test_preds.extend(p)
            test_gts.extend(batch["y"].cpu().numpy())
            test_seqs.extend(batch["seq"])
    exp_rec["predictions"] = test_preds
    exp_rec["ground_truth"] = test_gts

    experiment_data["hidden_dim"][hdim] = exp_rec

# ----------------------------- save --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
