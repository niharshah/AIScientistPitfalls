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

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
import pathlib
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
}


# ---------- Utility functions & metrics ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# -------------------------------------------------

# ---------- Load data (fallback to synthetic) ----------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dataset_available = pathlib.Path(DATA_ENV).exists()
if dataset_available:
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("SPR_BENCH not found. Creating synthetic data for demo.")

    def synth_split(n):
        seqs, labels = [], []
        shapes = ["▲", "●", "■"]
        colors = ["r", "g", "b"]
        for i in range(n):
            seq = " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            label = np.random.choice(["A", "B", "C"])
            seqs.append(seq)
            labels.append(label)
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth_split(n)}, split="train"
        )
# --------------------------------------------------------

# ---------- Text vectorisation ----------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])
vocab_size = len(vectorizer.vocabulary_)
print(f"Vocabulary size: {vocab_size}")


def vectorize(seqs: List[str]) -> np.ndarray:
    return vectorizer.transform(seqs).toarray().astype(np.float32)


X_train = vectorize(dsets["train"]["sequence"])
X_val = vectorize(dsets["dev"]["sequence"])
X_test = vectorize(dsets["test"]["sequence"])

# ---------- Label encoding ----------
labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

y_train = np.array([label2id[l] for l in dsets["train"]["label"]], dtype=np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], dtype=np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], dtype=np.int64)
num_classes = len(labels)
print(f"Number of classes: {num_classes}")

# ---------- DataLoader ----------
batch_size = 64
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=batch_size,
)


# ---------- Model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- Training ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    all_preds, all_tgts, all_seqs = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_tgts.extend(yb.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    sequences_val = dsets["dev"]["sequence"]
    acc = (np.array(all_preds) == np.array(all_tgts)).mean()
    cwa = color_weighted_accuracy(sequences_val, all_tgts, all_preds)
    swa = shape_weighted_accuracy(sequences_val, all_tgts, all_preds)
    comp = complexity_weighted_accuracy(sequences_val, all_tgts, all_preds)

    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
    )
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  validation_loss={val_loss:.4f}  "
        f"ACC={acc:.3f}  CWA={cwa:.3f}  SWA={swa:.3f}  CompWA={comp:.3f}"
    )

# ---------- Final evaluation on test ----------
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).to(device)
    test_logits = model(X_test_tensor)
    test_preds = test_logits.argmax(dim=1).cpu().numpy()
test_sequences = dsets["test"]["sequence"]
test_acc = (test_preds == y_test).mean()
test_cwa = color_weighted_accuracy(test_sequences, y_test, test_preds)
test_swa = shape_weighted_accuracy(test_sequences, y_test, test_preds)
test_comp = complexity_weighted_accuracy(test_sequences, y_test, test_preds)

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = y_test
experiment_data["SPR_BENCH"]["sequences"] = test_sequences
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "compwa": test_comp,
}

print(
    f"\nTest set —  ACC={test_acc:.3f}  CWA={test_cwa:.3f}  "
    f"SWA={test_swa:.3f}  CompWA={test_comp:.3f}"
)

# ---------- Save experiment data ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved experiment data to {os.path.join(working_dir, 'experiment_data.npy')}")
