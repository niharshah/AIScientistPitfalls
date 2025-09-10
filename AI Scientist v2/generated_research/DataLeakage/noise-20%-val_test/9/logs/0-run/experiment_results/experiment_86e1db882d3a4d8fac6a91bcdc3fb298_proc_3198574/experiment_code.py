import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time
import pathlib
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset, DatasetDict
import random


# ---------------- Utility / Reproducibility ---------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Data Loading ---------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr_bench = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr_bench.keys())

# ---------------- Vocabulary ---------------- #
all_chars = set()
for ex in spr_bench["train"]["sequence"]:
    all_chars.update(list(ex))
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
vocab_size = len(char2idx)
print(f"Vocab size: {vocab_size}")


def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        if ch in char2idx:
            vec[char2idx[ch]] += 1.0
    # optional normalisation: term-frequency
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec


def prepare_split(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]])
    labels = np.array(split["label"], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(labels)


X_train, y_train = prepare_split(spr_bench["train"])
X_dev, y_dev = prepare_split(spr_bench["dev"])
X_test, y_test = prepare_split(spr_bench["test"])

num_classes = int(max(y_train.max(), y_dev.max(), y_test.max()) + 1)
print(f"Number of classes: {num_classes}")

train_ds = TensorDataset(X_train, y_train)
dev_ds = TensorDataset(X_dev, y_dev)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)


# ---------------- Model ---------------- #
class CharBagLinear(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        return self.linear(x)


model = CharBagLinear(vocab_size, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

# ---------------- Experiment Data Store ---------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_acc": [],
            "val_acc": [],
            "RBA": [],  # placeholder Rule-Based Accuracy
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------------- Evaluation Function ---------------- #
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds_all, gts_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            preds_all.append(preds.cpu())
            gts_all.append(yb.cpu())
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    acc = correct / total
    avg_loss = loss_sum / total
    preds_all = torch.cat(preds_all)
    gts_all = torch.cat(gts_all)
    return acc, avg_loss, preds_all.numpy(), gts_all.numpy()


# ---------------- Training Loop ---------------- #
num_epochs = 10
start_time = time.time()
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss, epoch_correct, seen = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        epoch_loss += loss.item() * yb.size(0)
        epoch_correct += (preds == yb).sum().item()
        seen += yb.size(0)

    train_acc = epoch_correct / seen
    train_loss = epoch_loss / seen

    val_acc, val_loss, _, _ = evaluate(dev_loader)

    # Placeholder RBA (no symbolic rule extraction yet)
    rba = 0.0

    # Record
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["RBA"].append(rba)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )

# ---------------- Final Test Evaluation ---------------- #
test_acc, test_loss, test_preds, test_gts = evaluate(test_loader)
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# ---------------- Save Experiment Data ---------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")
print(f"Total run time: {time.time() - start_time:.2f}s")
