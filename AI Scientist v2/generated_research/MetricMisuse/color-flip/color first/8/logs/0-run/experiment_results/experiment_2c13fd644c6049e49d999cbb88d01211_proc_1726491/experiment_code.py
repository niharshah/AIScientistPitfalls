import os, pathlib, random, itertools, time, warnings, numpy as np
from collections import Counter
from typing import Dict, List

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------- utility functions -------------------------
def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def color_weighted_accuracy(seqs, y, g):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y, g):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / (sum(w) or 1)


def harmonic_mean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ---------------- data loading ------------------------------
def load_real_spr(root: pathlib.Path):
    from datasets import load_dataset

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = {}
    for split in ("train.csv", "dev.csv", "test.csv"):
        d[split.split(".")[0]] = _load(split)
    return {k: list(v) for k, v in d.items()}


def create_synthetic_spr(n_train=400, n_dev=100, n_test=100, seq_len=8):
    shapes = list("ABCD")
    colors = list("1234")

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
        )

    def make_label(seq):
        return Counter(t[0] for t in seq.split()).most_common(1)[0][0]

    def make_split(n):
        return [
            {"id": i, "sequence": (s := make_seq()), "label": make_label(s)}
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


# ---------------- token mapping -----------------------------
def extract_tokens(split):
    for r in split:
        for tok in r["sequence"].split():
            yield tok


all_tokens = list(extract_tokens(spr_data["train"]))
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}


def token_vector(tok):
    return [shape2idx[tok[0]], color2idx[tok[1]]]


# ---------------- Dataset class -----------------------------
class SPRDataset(Dataset):
    def __init__(self, split_rows, seq2feat, lab_enc: LabelEncoder):
        self.seqs = [r["sequence"] for r in split_rows]
        self.x = np.stack([seq2feat(s) for s in self.seqs]).astype(np.float32)
        self.y = lab_enc.transform([r["label"] for r in split_rows]).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx]),
            "seq": self.seqs[idx],
        }


# ---------------- hyperparameter loop -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 15
cluster_grid = [4, 8, 16, 32]

le = LabelEncoder()
le.fit([r["label"] for r in spr_data["train"]])

experiment_data = {}
for k_clusters in cluster_grid:
    print(f"\n===== Training with {k_clusters} KMeans clusters =====")
    # fit kmeans on train token vectors
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans.fit(np.array([token_vector(t) for t in all_tokens]))

    def sequence_to_features(seq: str):
        toks = seq.split()
        clusters = kmeans.predict(np.array([token_vector(t) for t in toks]))
        hist = np.bincount(clusters, minlength=k_clusters) / len(toks)
        return np.concatenate(
            [hist, [count_shape_variety(seq), count_color_variety(seq)]]
        )

    # datasets
    train_ds = SPRDataset(spr_data["train"], sequence_to_features, le)
    dev_ds = SPRDataset(spr_data["dev"], sequence_to_features, le)
    test_ds = SPRDataset(spr_data["test"], sequence_to_features, le)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    # model
    model = nn.Sequential(
        nn.Linear(train_ds.x.shape[1], 32), nn.ReLU(), nn.Linear(32, len(le.classes_))
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # storage
    exp_key = f"k{k_clusters}"
    experiment_data[exp_key] = {
        "spr_bench": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    # training loop
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optim.zero_grad()
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optim.step()
            tr_loss += loss.item() * batch["y"].size(0)
        tr_loss /= len(train_ds)
        experiment_data[exp_key]["spr_bench"]["losses"]["train"].append((ep, tr_loss))
        # validation
        model.eval()
        val_loss = 0.0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                p = out.argmax(1).cpu().numpy()
                preds.extend(p)
                gts.extend(batch["y"].cpu().numpy())
                seqs.extend(batch["seq"])
        val_loss /= len(dev_ds)
        experiment_data[exp_key]["spr_bench"]["losses"]["val"].append((ep, val_loss))
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cshm = harmonic_mean(cwa, swa)
        experiment_data[exp_key]["spr_bench"]["metrics"]["val"].append(
            (ep, cwa, swa, cshm)
        )
        print(
            f"Ep{ep:02d} k={k_clusters} | val_loss {val_loss:.4f} | CWA {cwa:.3f} SWA {swa:.3f} CSHM {cshm:.3f}"
        )
    # test predictions
    model.eval()
    t_preds = []
    t_gts = []
    t_seqs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logit = model(batch["x"])
            p = logit.argmax(1).cpu().numpy()
            t_preds.extend(p)
            t_gts.extend(batch["y"].cpu().numpy())
            t_seqs.extend(batch["seq"])
    experiment_data[exp_key]["spr_bench"]["predictions"] = t_preds
    experiment_data[exp_key]["spr_bench"]["ground_truth"] = t_gts

# ---------------- save ---------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", working_dir)
