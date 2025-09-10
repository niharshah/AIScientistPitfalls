# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

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

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Hyper-params ---------------- #
BATCH_SIZE = 256
VAL_BATCH = 512
LR = 1e-2
EPOCHS = 10
RULE_TOP_K = 1  # characters per class that constitute the symbolic “rule”


# ---------------- Dataset loading ---------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr_bench = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr_bench.keys())

# ---------------- Vocabulary ---------------- #
all_chars = set()
for seq in spr_bench["train"]["sequence"]:
    all_chars.update(seq)
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx)
print(f"Vocab size = {vocab_size}")


def seq_to_vec(seq: str) -> np.ndarray:
    """Bag-of-chars (frequency)."""
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[char2idx[ch]] += 1.0
    if len(seq) > 0:  # normalise
        vec /= len(seq)
    return vec


def prepare_split(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]])
    y = np.array(split["label"], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


X_train, y_train = prepare_split(spr_bench["train"])
X_dev, y_dev = prepare_split(spr_bench["dev"])
X_test, y_test = prepare_split(spr_bench["test"])

num_classes = int(max(y_train.max(), y_dev.max(), y_test.max()) + 1)
print(f"Number of classes: {num_classes}")

# ---------------- DataLoaders ---------------- #
train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=VAL_BATCH)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=VAL_BATCH)


# ---------------- Model ---------------- #
class CharBagLinear(nn.Module):
    def __init__(self, in_dim: int, num_cls: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        return self.linear(x)


model = CharBagLinear(vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- Experiment store ---------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "RBA": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------- Helper: evaluation ---------------- #
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return correct / total, loss_sum / total


# ---------------- Helper: Rule-Based Accuracy (RBA) ---------------- #
def compute_rule_accuracy(loader):
    """
    Build a trivial symbolic rule set: for each class, pick the top-K
    characters with highest weight. Predict by counting occurrences of those
    characters and picking the class with max count.
    """
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()  # [C, V]
    # top K indices per class
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]  # [C, K]

    total, correct = 0, 0
    for xb, yb in loader:
        seq_vectors = xb.numpy()  # frequencies
        # revert to counts to avoid floating
        counts = (seq_vectors * 1000).astype(int)  # arbitrary scaling
        preds = []
        for count_vec in counts:
            votes = []
            for cls in range(num_classes):
                votes.append(count_vec[top_idx[cls]].sum())
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total


# ---------------- Training loop ---------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_correct, seen = 0.0, 0, 0
    start_t = time.time()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        running_loss += loss.item() * yb.size(0)
        running_correct += (preds == yb).sum().item()
        seen += yb.size(0)

    train_acc = running_correct / seen
    train_loss = running_loss / seen
    val_acc, val_loss = evaluate(val_loader)
    rba = compute_rule_accuracy(val_loader)

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["RBA"].append(rba)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
        f"RBA={rba:.3f} | "
        f"epoch_time={time.time()-start_t:.1f}s"
    )

# ---------------- Final test evaluation ---------------- #
test_acc, test_loss = evaluate(test_loader)
rba_test = compute_rule_accuracy(test_loader)
print(f"\nTest set: loss={test_loss:.4f}, acc={test_acc:.3f}, RBA={rba_test:.3f}")

# Store final predictions for interpretability
model.eval()
all_preds, all_gts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(1).cpu()
        all_preds.append(preds)
        all_gts.append(yb)
experiment_data["SPR_BENCH"]["predictions"] = torch.cat(all_preds).numpy()
experiment_data["SPR_BENCH"]["ground_truth"] = torch.cat(all_gts).numpy()

# ---------------- Save everything ---------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")
