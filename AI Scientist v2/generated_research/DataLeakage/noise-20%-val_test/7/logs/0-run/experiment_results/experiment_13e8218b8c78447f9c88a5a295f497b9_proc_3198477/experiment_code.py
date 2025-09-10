import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict
import pathlib
from collections import Counter

# -------- GPU / device handling ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------- utility to load SPR_BENCH -------------------------------------------
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


# -------- data path ------------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
assert DATA_PATH.exists(), f"Dataset path {DATA_PATH} not found."

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------- build vocabulary -----------------------------------------------------
all_chars = Counter()
for seq in spr["train"]["sequence"]:
    all_chars.update(seq)
vocab = sorted(all_chars.keys())
char2idx = {c: i for i, c in enumerate(vocab)}
V = len(vocab)
print(f"Vocab size: {V}")

# -------- encode labels --------------------------------------------------------
le = LabelEncoder()
all_labels = le.fit_transform(spr["train"]["label"])
num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")


# -------- vectorisation helpers -----------------------------------------------
def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(V, dtype=np.float32)
    for ch in seq:
        if ch in char2idx:
            vec[char2idx[ch]] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()  # length-norm for stability
    return vec


def encode_split(split):
    X = np.vstack([seq_to_vec(s) for s in spr[split]["sequence"]])
    y = le.transform(spr[split]["label"])
    return X, y


X_train, y_train = encode_split("train")
X_dev, y_dev = encode_split("dev")
X_test, y_test = encode_split("test")


# -------- torch Dataset --------------------------------------------------------
class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


batch_size = 128
train_dl = DataLoader(VecDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(VecDataset(X_dev, y_dev), batch_size=batch_size)


# -------- model ----------------------------------------------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


model = LogReg(V, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# -------- experiment data store -----------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------- training loop --------------------------------------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    # --- train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in train_dl:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["y"].size(0)
        pred = logits.argmax(1)
        correct += (pred == batch["y"]).sum().item()
        total += batch["y"].size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # --- validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch in dev_dl:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            val_loss += loss.item() * batch["y"].size(0)
            pred = logits.argmax(1)
            val_correct += (pred == batch["y"]).sum().item()
            val_total += batch["y"].size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")

# -------- rule extraction ------------------------------------------------------
W = model.linear.weight.detach().cpu().numpy()  # (C, V)
b = model.linear.bias.detach().cpu().numpy()  # (C,)


def rule_predict(vec: np.ndarray):
    scores = W @ vec + b
    return scores.argmax()


# -------- evaluate on test split ----------------------------------------------
test_preds, rule_preds = [], []
for vec in X_test:
    p = rule_predict(vec)
    rule_preds.append(p)
    test_preds.append(p)  # model+rule identical

test_preds = np.array(test_preds)
rule_preds = np.array(rule_preds)
y_test_np = y_test

accuracy = (test_preds == y_test_np).mean()
fidelity = (rule_preds == test_preds).mean()  # should be 1.0
FAGM = np.sqrt(accuracy * fidelity)

print(f"\nTest accuracy = {accuracy:.4f}")
print(f"Rule fidelity  = {fidelity:.4f}")
print(f"FAGM           = {FAGM:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = y_test_np
experiment_data["SPR_BENCH"]["final_metrics"] = {
    "accuracy": accuracy,
    "fidelity": fidelity,
    "FAGM": FAGM,
}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# -------- quick interpretability demo: top-5 chars per class -------------------
for c, cls_name in enumerate(le.classes_):
    top5 = np.argsort(-W[c])[:5]
    chars = [vocab[i] for i in top5]
    print(f"Class {cls_name}: top-5 positive chars -> {chars}")
