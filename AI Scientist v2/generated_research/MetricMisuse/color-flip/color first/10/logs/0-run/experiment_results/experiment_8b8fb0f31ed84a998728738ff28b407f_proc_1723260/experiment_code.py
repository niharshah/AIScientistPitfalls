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

import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_CompWA": [], "val_CompWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------- Utility identical to snippet -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ----------------------------- Data ----------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

# extract glyph tokens
train_sequences = spr["train"]["sequence"]
dev_sequences = spr["dev"]["sequence"]
test_sequences = spr["test"]["sequence"]  # unused baseline


def get_tokens(seqs):
    tokens = []
    for s in seqs:
        tokens.extend(s.strip().split())
    return tokens


all_tokens = get_tokens(train_sequences)
shapes = [t[0] for t in all_tokens]
colors = [t[1] for t in all_tokens]
shape_le = LabelEncoder().fit(shapes)
color_le = LabelEncoder().fit(colors)

token_vectors = np.stack(
    [shape_le.transform(shapes), color_le.transform(colors)], axis=1
)

# -------------------------- Clustering -------------------------------
k = 8
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
kmeans.fit(token_vectors)


def sequence_to_histogram(seq):
    vec = np.zeros(k, dtype=np.float32)
    for tok in seq.strip().split():
        if len(tok) < 2:
            continue
        s_id = shape_le.transform([tok[0]])[0]
        c_id = color_le.transform([tok[1]])[0]
        label = kmeans.predict([[s_id, c_id]])[0]
        vec[label] += 1.0
    return vec


X_train = np.stack([sequence_to_histogram(s) for s in train_sequences])
X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences])

y_train = np.array(spr["train"]["label"], dtype=np.float32)
y_dev = np.array(spr["dev"]["label"], dtype=np.float32)

# -------------------------- Torch Model ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleFF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = SimpleFF(k).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

batch_size = 512
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)

# ----------------------------- Train ---------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- validation ----
    model.eval()
    val_loss, preds, truths, seqs_collected = 0.0, [], [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            preds.extend((torch.sigmoid(out) > 0.5).cpu().numpy().astype(int).tolist())
            truths.extend(yb.cpu().numpy().astype(int).tolist())
    val_loss /= len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    comp_wa = complexity_weighted_accuracy(dev_sequences, truths, preds)
    train_comp_wa = 0.0  # quick estimate on last batch optional
    experiment_data["SPR_BENCH"]["metrics"]["val_CompWA"].append(comp_wa)
    experiment_data["SPR_BENCH"]["metrics"]["train_CompWA"].append(None)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_CompWA = {comp_wa:.4f}"
    )

# ------------------ Final additional metrics -------------------------
cwa = color_weighted_accuracy(dev_sequences, y_dev, preds)
swa = shape_weighted_accuracy(dev_sequences, y_dev, preds)
print(f"Dev CWA: {cwa:.4f}, Dev SWA: {swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = truths
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
