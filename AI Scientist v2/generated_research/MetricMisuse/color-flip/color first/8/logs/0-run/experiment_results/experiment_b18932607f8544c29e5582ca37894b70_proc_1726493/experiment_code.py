import os, pathlib, random, itertools, time, math
import numpy as np
from collections import Counter
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------ I/O & experiment dict -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"weight_decay_tuning": {}}  # will be filled per run


# ------------------ utility functions -----------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_color_variety(s) for s in sequences]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def shape_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_shape_variety(s) for s in sequences]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def harmonic_mean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ------------------ data loading (real or synthetic) --------------
def load_real_spr(root: pathlib.Path):
    from datasets import load_dataset

    d = {}
    for split in ("train", "dev", "test"):
        d[split] = list(
            load_dataset(
                "csv",
                data_files=str(root / f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )
        )
    return d


def create_synthetic_spr(n_train=400, n_dev=100, n_test=100, seq_len=8):
    shapes = list("ABCD")
    colors = list("1234")

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
        )

    def make_label(seq):
        return Counter(t[0] for t in seq.split()).most_common(1)[0][0]

    def ms(n):
        return [
            {"id": i, "sequence": (s := make_seq()), "label": make_label(s)}
            for i in range(n)
        ]

    return {"train": ms(n_train), "dev": ms(n_dev), "test": ms(n_test)}


try:
    DATA_PATH = pathlib.Path("SPR_BENCH")
    spr_data = load_real_spr(DATA_PATH)
except Exception:
    print("Real SPR_BENCH not found – using synthetic data.")
    spr_data = create_synthetic_spr()
print({k: len(v) for k, v in spr_data.items()})


# ------------------ tokenisation & kmeans feats -------------------
def extract_tokens(split):
    for r in split:
        for t in r["sequence"].split():
            yield t


all_tokens = list(extract_tokens(spr_data["train"]))
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}


def token_vector(t):
    return [shape2idx[t[0]], color2idx[t[1]]]


token_vecs = np.array([token_vector(t) for t in all_tokens])
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(token_vecs)


def sequence_to_features(seq: str) -> np.ndarray:
    tokens = seq.split()
    clusters = kmeans.predict(np.array([token_vector(t) for t in tokens]))
    hist = np.bincount(clusters, minlength=n_clusters) / len(tokens)
    return np.concatenate([hist, [count_shape_variety(seq), count_color_variety(seq)]])


# ------------------ dataset / dataloaders -------------------------
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
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------ training routine ------------------------------
def run_experiment(weight_decay: float, epochs: int = 15):
    key = f"wd_{weight_decay}"
    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "weight_decay": weight_decay,
    }
    model = nn.Sequential(
        nn.Linear(train_ds.x.shape[1], 32), nn.ReLU(), nn.Linear(32, n_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for b in train_loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            optim.zero_grad()
            out = model(b["x"])
            loss = criterion(out, b["y"])
            loss.backward()
            optim.step()
            tr_loss += loss.item() * b["y"].size(0)
        tr_loss /= len(train_ds)
        run_data["losses"]["train"].append((ep, tr_loss))
        # validation
        model.eval()
        val_loss = 0.0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for b in dev_loader:
                b = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in b.items()
                }
                out = model(b["x"])
                loss = criterion(out, b["y"])
                val_loss += loss.item() * b["y"].size(0)
                p = out.argmax(1).cpu().numpy()
                preds.extend(p)
                gts.extend(b["y"].cpu().numpy())
                seqs.extend(b["seq"])
        val_loss /= len(dev_ds)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cshm = harmonic_mean(cwa, swa)
        run_data["losses"]["val"].append((ep, val_loss))
        run_data["metrics"]["val"].append((ep, cwa, swa, cshm))
        print(
            f"[wd={weight_decay}] Ep{ep} val_loss {val_loss:.4f} CWA {cwa:.3f} SWA {swa:.3f} CSHM {cshm:.3f}"
        )
    # test predictions
    model.eval()
    t_preds = []
    t_gts = []
    with torch.no_grad():
        for b in test_loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            p = model(b["x"]).argmax(1).cpu().numpy()
            t_preds.extend(p)
            t_gts.extend(b["y"].cpu().numpy())
    run_data["predictions"] = t_preds
    run_data["ground_truth"] = t_gts
    experiment_data["weight_decay_tuning"][key] = run_data


# ------------------ hyperparameter grid search --------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
for wd in weight_decays:
    run_experiment(wd)

# ------------------ save ------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)
