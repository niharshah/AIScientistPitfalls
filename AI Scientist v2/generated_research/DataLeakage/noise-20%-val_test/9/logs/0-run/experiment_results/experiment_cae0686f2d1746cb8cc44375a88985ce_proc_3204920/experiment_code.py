# Hyper-parameter Tuning : BATCH_SIZE
# self-contained single-file script
import os, time, pathlib
from typing import Dict, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Fixed hyper-params ---------------- #
VAL_BATCH = 512
LR = 1e-2
EPOCHS = 10
RULE_TOP_K = 1
BATCH_SIZE_CANDIDATES = [32, 64, 128, 256, 512]  # tunable list


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
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[char2idx[ch]] += 1.0
    if len(seq) > 0:
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


# ---------------- Model class ---------------- #
class CharBagLinear(nn.Module):
    def __init__(self, in_dim: int, num_cls: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        return self.linear(x)


# ---------------- Helper functions ---------------- #
criterion = nn.CrossEntropyLoss()


def evaluate(model: nn.Module, loader: DataLoader):
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


def compute_rule_accuracy(model: nn.Module, loader: DataLoader):
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()  # [C, V]
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]
    total, correct = 0, 0
    for xb, yb in loader:
        seq_vecs = xb.numpy()
        counts = (seq_vecs * 1000).astype(int)
        preds = []
        for count_vec in counts:
            votes = [count_vec[top_idx[cls]].sum() for cls in range(num_classes)]
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total


# ---------------- Experiment store ---------------- #
experiment_data = {
    "BATCH_SIZE": {
        "SPR_BENCH": {
            "batch_sizes": [],
            "metrics": {"train_acc": [], "val_acc": [], "test_acc": []},
            "losses": {"train": [], "val": [], "test": []},
            "RBA_val": [],
            "RBA_test": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------------- Hyper-parameter tuning loop ---------------- #
best_val_acc, best_bs = -1, None
for bs in BATCH_SIZE_CANDIDATES:
    print(f"\n===== Training with BATCH_SIZE = {bs} =====")
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=bs, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=VAL_BATCH)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=VAL_BATCH)

    # Model, optimiser
    model = CharBagLinear(vocab_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epoch_train_acc, epoch_val_acc, epoch_train_loss, epoch_val_loss, epoch_rba = (
        [],
        [],
        [],
        [],
        [],
    )
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

        t_acc = running_correct / seen
        t_loss = running_loss / seen
        v_acc, v_loss = evaluate(model, val_loader)
        rba_val = compute_rule_accuracy(model, val_loader)

        epoch_train_acc.append(t_acc)
        epoch_val_acc.append(v_acc)
        epoch_train_loss.append(t_loss)
        epoch_val_loss.append(v_loss)
        epoch_rba.append(rba_val)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={t_loss:.4f}, train_acc={t_acc:.3f} | "
            f"val_loss={v_loss:.4f}, val_acc={v_acc:.3f} | "
            f"RBA={rba_val:.3f} | "
            f"time={time.time()-start_t:.1f}s"
        )

    # Test evaluation
    test_acc, test_loss = evaluate(model, test_loader)
    rba_test = compute_rule_accuracy(model, test_loader)
    print(f"Finished BATCH_SIZE={bs}: test_acc={test_acc:.3f}, RBA_test={rba_test:.3f}")

    # Store results
    ed = experiment_data["BATCH_SIZE"]["SPR_BENCH"]
    ed["batch_sizes"].append(bs)
    ed["metrics"]["train_acc"].append(epoch_train_acc)
    ed["metrics"]["val_acc"].append(epoch_val_acc)
    ed["metrics"]["test_acc"].append(test_acc)
    ed["losses"]["train"].append(epoch_train_loss)
    ed["losses"]["val"].append(epoch_val_loss)
    ed["losses"]["test"].append(test_loss)
    ed["RBA_val"].append(epoch_rba)
    ed["RBA_test"].append(rba_test)

    # Save predictions & ground truth for this batch size
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_gts.append(yb)
    ed["predictions"].append(torch.cat(all_preds).numpy())
    ed["ground_truth"].append(torch.cat(all_gts).numpy())

    # Track best model by validation accuracy
    if max(epoch_val_acc) > best_val_acc:
        best_val_acc = max(epoch_val_acc)
        best_bs = bs

print(
    f"\nBest validation accuracy {best_val_acc:.3f} achieved with BATCH_SIZE = {best_bs}"
)

# ---------------- Save experiment ---------------- #
np.save("experiment_data.npy", experiment_data)
print("All experiment data saved to experiment_data.npy")
