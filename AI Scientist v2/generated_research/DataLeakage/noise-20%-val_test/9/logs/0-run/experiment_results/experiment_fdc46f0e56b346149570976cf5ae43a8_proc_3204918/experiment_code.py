# -----------------------------------------------
# Hyper-parameter tuning: EPOCHS (single-file run)
# -----------------------------------------------
import os, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict
from typing import Dict

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Hyper-params ---------------- #
BATCH_SIZE = 256
VAL_BATCH = 512
LR = 1e-2
MAX_EPOCHS = 50  # upper bound for training epochs
PATIENCE = 5  # early-stopping patience
RULE_TOP_K = 1  # used by rule-based accuracy


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

# ---------------- Vocabulary + helpers ---------------- #
all_chars = set("".join(spr_bench["train"]["sequence"]))
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx)
print("Vocab size =", vocab_size)


def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[char2idx[ch]] += 1.0
    if len(seq):
        vec /= len(seq)
    return vec


def prepare_split(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]], dtype=np.float32)
    y = np.array(split["label"], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


X_train, y_train = prepare_split(spr_bench["train"])
X_dev, y_dev = prepare_split(spr_bench["dev"])
X_test, y_test = prepare_split(spr_bench["test"])
num_classes = int(max(y_train.max(), y_dev.max(), y_test.max()) + 1)
print("Number of classes:", num_classes)

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
experiment_data: Dict = {
    "epoch_tuning": {
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "test_acc": None},
            "losses": {"train": [], "val": [], "test": None},
            "RBA": [],
            "predictions": [],
            "ground_truth": [],
            "best_epoch": None,
        }
    }
}


# ---------------- Helpers ---------------- #
def evaluate(loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            tot += yb.size(0)
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return correct / tot, loss_sum / tot


def compute_rule_accuracy(loader):
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]
    tot, correct = 0, 0
    for xb, yb in loader:
        freq = xb.numpy()
        counts = (freq * 1000).astype(int)
        preds = []
        for vec in counts:
            votes = [vec[top_idx[c]].sum() for c in range(num_classes)]
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        correct += (preds == yb).sum().item()
        tot += yb.size(0)
    return correct / tot


# ---------------- Training with early-stopping ---------------- #
best_val, best_state, patience_left = float("inf"), None, PATIENCE
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss, epoch_correct, seen = 0.0, 0, 0
    t0 = time.time()
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
    val_acc, val_loss = evaluate(val_loader)
    rba = compute_rule_accuracy(val_loader)

    # Logging
    ed = experiment_data["epoch_tuning"]["SPR_BENCH"]
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["RBA"].append(rba)

    # Early stop check
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        ed["best_epoch"] = epoch
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left == 0:
            print(f"Early stopping at epoch {epoch}")
            break

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
        f"RBA={rba:.3f} | time={time.time()-t0:.1f}s"
    )

# ---------------- Final test evaluation ---------------- #
# Load best parameters
if best_state is not None:
    model.load_state_dict(best_state)

test_acc, test_loss = evaluate(test_loader)
rba_test = compute_rule_accuracy(test_loader)
print(f"\nTest: loss={test_loss:.4f} acc={test_acc:.3f} RBA={rba_test:.3f}")

# Save final results
ed = experiment_data["epoch_tuning"]["SPR_BENCH"]
ed["metrics"]["test_acc"] = test_acc
ed["losses"]["test"] = test_loss
ed["RBA"].append(rba_test)  # append test-set rba

# Store predictions for interpretability
model.eval()
preds_all, gts_all = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        preds_all.append(logits.argmax(1).cpu())
        gts_all.append(yb)
ed["predictions"] = torch.cat(preds_all).numpy()
ed["ground_truth"] = torch.cat(gts_all).numpy()

# ---------------- Save everything ---------------- #
np.save("experiment_data.npy", experiment_data)
print("\nExperiment data saved to experiment_data.npy")
