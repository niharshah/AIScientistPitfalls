import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# ---------------- SETUP ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------- DATA LOADING -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


# allow env override for dataset path
data_root = pathlib.Path(
    os.environ.get("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
dsets = load_spr_bench(data_root)
print({k: len(v) for k, v in dsets.items()})


# ---------------- N-GRAM VECTORIZER ----------------
def build_vocab(seqs: List[str]):
    unis, bis = set(), set()
    for s in seqs:
        unis.update(s)
        bis.update([s[i : i + 2] for i in range(len(s) - 1)])
    vocab = sorted(unis) + sorted(bis)
    return {tok: i for i, tok in enumerate(vocab)}


def vectorise(seq: str, idx: Dict[str, int]) -> np.ndarray:
    v = np.zeros(len(idx), dtype=np.float32)
    for c in seq:
        if c in idx:
            v[idx[c]] += 1.0
    for i in range(len(seq) - 1):
        bg = seq[i : i + 2]
        if bg in idx:
            v[idx[bg]] += 1.0
    return v


vocab_idx = build_vocab(dsets["train"]["sequence"])
num_feats = len(vocab_idx)
print(f"Feature size: {num_feats}")

labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Classes: {labels}")


def encode_split(split):
    X = np.stack([vectorise(s, vocab_idx) for s in dsets[split]["sequence"]])
    y = np.array([label2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y


X_train, y_train = encode_split("train")
X_dev, y_dev = encode_split("dev")
X_test, y_test = encode_split("test")


class NgramDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.X[idx]), "y": torch.tensor(self.y[idx])}


batch_size = 128
train_loader = DataLoader(
    NgramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=batch_size)


# ------------------ MODEL --------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


model = LogReg(num_feats, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- EXPERIMENT STORE -----------------------
experiment_data = {
    "epochs": {  # hyper-parameter tuned
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ---------------- EVALUATION FN --------------------
def evaluate(loader):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            preds = logits.argmax(1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            all_logits.append(logits.cpu())
    return correct / total, loss_sum / total, torch.cat(all_logits)


# ---------------- TRAINING LOOP --------------------
max_epochs = 50  # hyper-parameter upper bound
patience = 5  # early-stopping patience
best_val_loss = float("inf")
epochs_no_improve = 0
top_k = 10

for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = correct = seen = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["y"].size(0)
        correct += (logits.argmax(1) == batch["y"]).sum().item()
        seen += batch["y"].size(0)

    train_loss, train_acc = running_loss / seen, correct / seen
    val_acc, val_loss, _ = evaluate(dev_loader)

    # ---------- Rule Fidelity ----------
    with torch.no_grad():
        W = model.linear.weight.cpu().numpy()
        b = model.linear.bias.cpu().numpy()
        W_trunc = np.zeros_like(W)
        for c in range(num_classes):
            idxs = np.argsort(-np.abs(W[c]))[:top_k]
            W_trunc[c, idxs] = W[c, idxs]
        lin_full = torch.from_numpy(X_dev @ W.T + b)
        lin_trunc = torch.from_numpy(X_dev @ W_trunc.T + b)
        rule_fid = (lin_trunc.argmax(1) == lin_full.argmax(1)).float().mean().item()

    # log
    experiment_data["epochs"]["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["epochs"]["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["epochs"]["SPR_BENCH"]["metrics"]["rule_fidelity"].append(rule_fid)
    experiment_data["epochs"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["epochs"]["SPR_BENCH"]["losses"]["val"].append(val_loss)

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
    )

    # -------- Early Stopping ------------
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(
                f"No improvement for {patience} epochs, stopping early at epoch {epoch}"
            )
            break

# ------------------ TEST EVAL ----------------------
test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=batch_size)
test_acc, test_loss, test_logits = evaluate(test_loader)
experiment_data["epochs"]["SPR_BENCH"]["predictions"] = test_logits.argmax(1).numpy()
experiment_data["epochs"]["SPR_BENCH"]["ground_truth"] = y_test
print(f"Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")

# ---------------- SAVE RESULTS --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
