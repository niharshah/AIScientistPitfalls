import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim, copy
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import DatasetDict, load_dataset

# ----------------------- I/O & DEVICE -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------- DATA LOADING -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
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


# -------------------- N-GRAM VECTORIZER ---------------------
def build_vocab(seqs: List[str]):
    unis, bis = set(), set()
    for s in seqs:
        chars = list(s)
        unis.update(chars)
        bis.update([s[i : i + 2] for i in range(len(s) - 1)])
    vocab = sorted(list(unis)) + sorted(list(bis))
    return {t: i for i, t in enumerate(vocab)}


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

labels = sorted(list(set(dsets["train"]["label"])))
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
train_loader_full = DataLoader(
    NgramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=batch_size)


# ------------------- MODEL DEFINITION ----------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()

# ----------------- EXPERIMENT DATA DICT -------------------
experiment_data = {
    "adam_beta2": {
        "SPR_BENCH": {
            "beta2_values": [],
            "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test,
            "best_beta2": None,
        }
    }
}


# ------------------ TRAINING UTILITIES --------------------
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device).float() if k == "x" else v.to(device)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            _, preds = torch.max(logits, 1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            all_logits.append(logits.cpu())
    return correct / total, loss_sum / total, torch.cat(all_logits)


# ----------------- HYPERPARAMETER TUNING ------------------
beta2_grid = [0.98, 0.985, 0.99, 0.992, 0.995]
epochs = 10
best_val_acc = -1.0
best_state = None
best_beta2 = None
best_run_data = {}

for beta2 in beta2_grid:
    print(f"\n--- Training with beta2={beta2} ---")
    model = LogReg(num_feats, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, beta2))
    train_loader = train_loader_full  # reuse iterator
    run_train_acc, run_val_acc, run_rule_fid = [], [], []
    run_train_loss, run_val_loss = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device).float() if k == "x" else v.to(device)
                for k, v in batch.items()
            }
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
        top_k = 10
        W_trunc = np.zeros_like(W)
        for c in range(num_classes):
            idxs = np.argsort(-np.abs(W[c]))[:top_k]
            W_trunc[c, idxs] = W[c, idxs]
        lin_full = torch.from_numpy((X_dev @ W.T) + b)
        lin_trunc = torch.from_numpy((X_dev @ W_trunc.T) + b)
        rule_pred = torch.argmax(lin_trunc, 1)
        model_pred = torch.argmax(lin_full, 1)
        rule_fid = (rule_pred == model_pred).float().mean().item()

        run_train_acc.append(train_acc)
        run_val_acc.append(val_acc)
        run_rule_fid.append(rule_fid)
        run_train_loss.append(train_loss)
        run_val_loss.append(val_loss)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
        )

    # Store per-run data
    experiment_data["adam_beta2"]["SPR_BENCH"]["beta2_values"].append(beta2)
    experiment_data["adam_beta2"]["SPR_BENCH"]["metrics"]["train_acc"].append(
        run_train_acc
    )
    experiment_data["adam_beta2"]["SPR_BENCH"]["metrics"]["val_acc"].append(run_val_acc)
    experiment_data["adam_beta2"]["SPR_BENCH"]["metrics"]["rule_fidelity"].append(
        run_rule_fid
    )
    experiment_data["adam_beta2"]["SPR_BENCH"]["losses"]["train"].append(run_train_loss)
    experiment_data["adam_beta2"]["SPR_BENCH"]["losses"]["val"].append(run_val_loss)

    # Track best model
    if max(run_val_acc) > best_val_acc:
        best_val_acc = max(run_val_acc)
        best_state = copy.deepcopy(model.state_dict())
        best_beta2 = beta2
        best_run_data = {
            "train_acc": run_train_acc,
            "val_acc": run_val_acc,
            "rule_fidelity": run_rule_fid,
            "train_loss": run_train_loss,
            "val_loss": run_val_loss,
        }

# -------------------- FINAL EVALUATION --------------------
print(f"\nBest beta2={best_beta2} with val_acc={best_val_acc:.3f}")
best_model = LogReg(num_feats, num_classes).to(device)
best_model.load_state_dict(best_state)
test_acc, test_loss, test_logits = evaluate(best_model, test_loader)
print(f"Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")

experiment_data["adam_beta2"]["SPR_BENCH"]["best_beta2"] = best_beta2
experiment_data["adam_beta2"]["SPR_BENCH"]["best_run"] = best_run_data
experiment_data["adam_beta2"]["SPR_BENCH"]["predictions"] = torch.argmax(
    test_logits, 1
).numpy()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
