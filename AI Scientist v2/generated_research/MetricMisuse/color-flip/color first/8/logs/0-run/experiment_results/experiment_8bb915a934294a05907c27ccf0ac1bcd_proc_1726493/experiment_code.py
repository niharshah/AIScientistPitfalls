import os, pathlib, random, itertools, time
import numpy as np
from collections import Counter
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ------------------------------------------------------------------
# ----------  Utility functions copied / adapted from SPR.py -------
def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) or 1)


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) or 1)


def harmonic_mean(a, b, eps=1e-8):
    return 2 * a * b / (a + b + eps)


# ------------------------------------------------------------------
# ----------------- Data loading (with fallback) --------------------
def load_real_spr(root: pathlib.Path):
    try:
        from datasets import load_dataset, DatasetDict

        def _load(csv_name: str):
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
    except Exception as e:
        raise e


def create_synthetic_spr(n_train=400, n_dev=100, n_test=100, seq_len=8):
    shapes = list("ABCD")
    colors = list("1234")

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
        )

    def make_label(seq):
        shapes_in = [tok[0] for tok in seq.split()]
        return Counter(shapes_in).most_common(1)[0][0]

    def make_split(n):
        data = []
        for i in range(n):
            seq = make_seq()
            data.append({"id": i, "sequence": seq, "label": make_label(seq)})
        return data

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


# ------------------------------------------------------------------
# -------- tokenize & basic symbol mappings ------------------------
def extract_tokens(split_data):
    for row in split_data:
        for tok in row["sequence"].split():
            yield tok


all_tokens = list(extract_tokens(spr_data["train"]))
shapes = sorted({tok[0] for tok in all_tokens})
colors = sorted({tok[1] for tok in all_tokens})
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}


def token_vector(tok: str):
    return [shape2idx[tok[0]], color2idx[tok[1]]]


# ------------------------------------------------------------------
# --------------------- KMeans clustering --------------------------
token_vecs = np.array([token_vector(t) for t in all_tokens])
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(token_vecs)


def sequence_to_features(seq: str) -> np.ndarray:
    tokens = seq.split()
    clusters = kmeans.predict(np.array([token_vector(t) for t in tokens]))
    hist = np.bincount(clusters, minlength=n_clusters) / len(tokens)
    shape_var = count_shape_variety(seq)
    color_var = count_color_variety(seq)
    return np.concatenate([hist, [shape_var, color_var]])


# ------------------------------------------------------------------
# ------------- prepare tensors, labels, dataset class ---------------
le = LabelEncoder()
le.fit([row["label"] for row in spr_data["train"]])
n_classes = len(le.classes_)


class SPRDataset(Dataset):
    def __init__(self, split_rows):
        self.seqs = [r["sequence"] for r in split_rows]
        self.x = np.stack([sequence_to_features(s) for s in self.seqs]).astype(
            np.float32
        )
        self.y = le.transform([r["label"] for r in split_rows]).astype(np.int64)

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

# ------------------------------------------------------------------
# ------------- structure for mandatory experiment_data ------------
experiment_data = {"batch_size": {}}

# ------------------------------------------------------------------
# ----------------------- training loop for tuning -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_sizes = [16, 32, 64, 128, 256]
epochs = 15
for bs in batch_sizes:
    print("\n======== Training with batch_size =", bs, "========")
    exp_key = f"spr_bench_bs{bs}"
    experiment_data["batch_size"][exp_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # loaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)
    # model / opt
    model = nn.Sequential(
        nn.Linear(train_ds.x.shape[1], 32), nn.ReLU(), nn.Linear(32, n_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # training epochs
    for epoch in range(1, epochs + 1):
        model.train()
        ep_train_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            outputs = model(batch["x"])
            loss = criterion(outputs, batch["y"])
            loss.backward()
            optimizer.step()
            ep_train_loss += loss.item() * batch["y"].size(0)
        ep_train_loss /= len(train_ds)
        experiment_data["batch_size"][exp_key]["losses"]["train"].append(
            (epoch, ep_train_loss)
        )

        # validation
        model.eval()
        val_loss, preds, gts, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = model(batch["x"])
                loss = criterion(outputs, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                pred_labels = outputs.argmax(dim=1).cpu().numpy()
                preds.extend(pred_labels)
                gts.extend(batch["y"].cpu().numpy())
                seqs.extend(batch["seq"])
        val_loss /= len(dev_ds)
        experiment_data["batch_size"][exp_key]["losses"]["val"].append(
            (epoch, val_loss)
        )

        # metrics
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cshm = harmonic_mean(cwa, swa)
        experiment_data["batch_size"][exp_key]["metrics"]["val"].append(
            (epoch, cwa, swa, cshm)
        )
        print(
            f"Epoch {epoch:02d}: val_loss {val_loss:.4f} | CWA {cwa:.3f} SWA {swa:.3f} CSHM {cshm:.3f}"
        )

    # -------------- final test predictions ----------------
    model.eval()
    test_preds, test_gts, test_seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            p = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(p)
            test_gts.extend(batch["y"].cpu().numpy())
            test_seqs.extend(batch["seq"])
    experiment_data["batch_size"][exp_key]["predictions"] = test_preds
    experiment_data["batch_size"][exp_key]["ground_truth"] = test_gts

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", working_dir)
