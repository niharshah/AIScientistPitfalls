import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import DatasetDict, load_dataset

# -------------------- I/O --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- DATA -------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# --------------- n-gram vectoriser -----------------
def build_vocab(seqs: List[str]):
    unis, bis = set(), set()
    for s in seqs:
        unis.update(list(s))
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


train_seqs = dsets["train"]["sequence"]
vocab_idx = build_vocab(train_seqs)
num_feats = len(vocab_idx)
print("Feature size:", num_feats)

labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print("Classes:", labels)


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
train_loader_full = DataLoader(
    NgramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=batch_size)


# -------------------- MODEL ------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()


# --------------- EVALUATION FUNC -------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = correct = loss_sum = 0.0
    all_logits = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss_sum += loss.item() * batch["y"].size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == batch["y"]).sum().item()
        total += batch["y"].size(0)
        all_logits.append(logits.cpu())
    return correct / total, loss_sum / total, torch.cat(all_logits)


# -------------------- SEARCH -----------------------
epoch_grid = [10, 20, 30, 40, 50]
patience = 5
top_k = 10

best_overall_acc = -1.0
best_run_data = None
search_val_accs = []

for max_epochs in epoch_grid:
    print(f"\n=== Training with max_epochs={max_epochs} ===")
    model = LogReg(num_feats, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    run_metrics = {"train_acc": [], "val_acc": [], "rule_fidelity": []}
    run_losses = {"train": [], "val": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = seen = correct = 0.0
        for batch in train_loader_full:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == batch["y"]).sum().item()
            seen += batch["y"].size(0)
        train_loss = running_loss / seen
        train_acc = correct / seen

        val_acc, val_loss, _ = evaluate(model, dev_loader)

        # Rule fidelity
        W = model.linear.weight.detach().cpu().numpy()
        b = model.linear.bias.detach().cpu().numpy()
        W_trunc = np.zeros_like(W)
        for c in range(num_classes):
            idxs = np.argsort(-np.abs(W[c]))[:top_k]
            W_trunc[c, idxs] = W[c, idxs]
        lin_full = torch.from_numpy((X_dev @ W.T) + b)
        lin_trunc = torch.from_numpy((X_dev @ W_trunc.T) + b)
        rule_pred = torch.argmax(lin_trunc, 1)
        model_pred = torch.argmax(lin_full, 1)
        rule_fid = (rule_pred == model_pred).float().mean().item()

        run_metrics["train_acc"].append(train_acc)
        run_metrics["val_acc"].append(val_acc)
        run_metrics["rule_fidelity"].append(rule_fid)
        run_losses["train"].append(train_loss)
        run_losses["val"].append(val_loss)

        print(
            f"Epoch {epoch:02d}/{max_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"rule_fid={rule_fid:.3f}"
        )

        # ------ Early stopping ------
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # reload best state for this run
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_val_acc, _, _ = evaluate(model, dev_loader)
    search_val_accs.append(best_val_acc)

    print(f"Run finished. Best epoch={best_epoch}, best val_acc={best_val_acc:.3f}")

    if best_val_acc > best_overall_acc:
        best_overall_acc = best_val_acc
        best_run_data = {
            "model_state": {k: v.clone() for k, v in best_state.items()},
            "metrics": run_metrics,
            "losses": run_losses,
            "max_epochs": max_epochs,
            "best_epoch": best_epoch,
        }

# ------------------- FINAL TEST --------------------
print(
    f"\nSelecting model from max_epochs={best_run_data['max_epochs']} "
    f"epoch={best_run_data['best_epoch']} with val_acc={best_overall_acc:.3f}"
)
best_model = LogReg(num_feats, num_classes).to(device)
best_model.load_state_dict(
    {k: v.to(device) for k, v in best_run_data["model_state"].items()}
)

test_acc, test_loss, test_logits = evaluate(best_model, test_loader)
print(f"Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")

# --------------- SAVE EXPERIMENT DATA --------------
experiment_data = {
    "num_epochs": {
        "SPR_BENCH": {
            "search_vals": np.array(epoch_grid),
            "search_val_accs": np.array(search_val_accs),
            "metrics": best_run_data["metrics"],
            "losses": best_run_data["losses"],
            "predictions": torch.argmax(test_logits, 1).numpy(),
            "ground_truth": y_test,
        }
    }
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
