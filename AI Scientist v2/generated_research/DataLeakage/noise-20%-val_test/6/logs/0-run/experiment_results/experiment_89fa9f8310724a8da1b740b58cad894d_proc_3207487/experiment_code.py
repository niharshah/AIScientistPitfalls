import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from typing import List, Dict

# ------------------------------ EXP STORE ------------------------------
experiment_data = {"max_n_gram_length": {"SPR_BENCH": {}}}  # per value filled below

# ------------------------------ SETUP ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------ DATA ----------------------------------
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


# --------------------- N-GRAM HELPER FUNCS ----------------------------
def build_vocab(seqs: List[str], max_n: int) -> Dict[str, int]:
    ngram_sets = [set() for _ in range(max_n)]
    for s in seqs:
        for n in range(1, max_n + 1):
            if len(s) < n:
                continue
            ngram_sets[n - 1].update(s[i : i + n] for i in range(len(s) - n + 1))
    vocab = []
    for n in range(1, max_n + 1):
        vocab.extend(sorted(list(ngram_sets[n - 1])))
    return {tok: i for i, tok in enumerate(vocab)}


def vectorise(seq: str, idx: Dict[str, int], max_n: int) -> np.ndarray:
    v = np.zeros(len(idx), dtype=np.float32)
    for n in range(1, max_n + 1):
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            ng = seq[i : i + n]
            if ng in idx:
                v[idx[ng]] += 1.0
    return v


# ---------------------- LABEL HANDLING --------------------------------
labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print("Classes:", labels)


# --------------------- DATASET CLASS ----------------------------------
class NgramDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.X[idx]), "y": torch.tensor(self.y[idx])}


# --------------------- MODEL ------------------------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# -------------------- EVALUATION FUNC ---------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    logits_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            _, preds = torch.max(logits, 1)
            tot += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            logits_all.append(logits.cpu())
    return correct / tot, loss_sum / tot, torch.cat(logits_all)


# ------------------ HYPERPARAM SWEEP ----------------------------------
max_n_values = [1, 2, 3]  # candidate n-gram lengths
epochs = 10
batch_size = 128
top_k = 10  # for rule fidelity
best_val_acc, best_artifact = -1, None

for n_val in max_n_values:
    print(f"\n=== Training with max_n_gram_length = {n_val} ===")
    # Build vocab and vectorise
    vocab_idx = build_vocab(dsets["train"]["sequence"], n_val)
    num_feats = len(vocab_idx)
    print(f"Feature size: {num_feats}")

    def encode(split):
        X = np.stack([vectorise(s, vocab_idx, n_val) for s in dsets[split]["sequence"]])
        y = np.array([label2id[l] for l in dsets[split]["label"]], dtype=np.int64)
        return X, y

    X_train, y_train = encode("train")
    X_dev, y_dev = encode("dev")
    X_test, y_test = encode("test")

    train_loader = DataLoader(
        NgramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=batch_size)

    # fresh model
    model = LogReg(num_feats, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # prepare storage
    experiment_data["max_n_gram_length"]["SPR_BENCH"][f"n={n_val}"] = {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
    }
    store = experiment_data["max_n_gram_length"]["SPR_BENCH"][f"n={n_val}"]

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, seen, correct = 0.0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * batch["y"].size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == batch["y"]).sum().item()
            seen += batch["y"].size(0)
        train_acc, train_loss = correct / seen, run_loss / seen
        val_acc, val_loss, _ = evaluate(model, dev_loader, criterion)

        # rule fidelity
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

        # store
        store["metrics"]["train_acc"].append(train_acc)
        store["metrics"]["val_acc"].append(val_acc)
        store["metrics"]["rule_fidelity"].append(rule_fid)
        store["losses"]["train"].append(train_loss)
        store["losses"]["val"].append(val_loss)

        print(
            f"epoch {epoch:02d} | train_acc {train_acc:.3f} val_acc {val_acc:.3f} rule_fid {rule_fid:.3f}"
        )

    # keep best model by val accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_artifact = {
            "n_val": n_val,
            "model_state": model.state_dict(),
            "vocab_idx": vocab_idx,
            "X_test": X_test,
            "y_test": y_test,
        }

# ----------------------- FINAL TEST EVAL ------------------------------
best_n = best_artifact["n_val"]
print(f"\nBest max_n_gram_length on dev = {best_n} (val_acc={best_val_acc:.3f})")

# recreate model for test eval (weights already saved)
best_vocab = best_artifact["vocab_idx"]
best_model = LogReg(len(best_vocab), num_classes).to(device)
best_model.load_state_dict(best_artifact["model_state"])
criterion = nn.CrossEntropyLoss()
test_loader = DataLoader(
    NgramDataset(best_artifact["X_test"], best_artifact["y_test"]),
    batch_size=batch_size,
)
test_acc, test_loss, test_logits = evaluate(best_model, test_loader, criterion)
print(f"Test_acc={test_acc:.3f} Test_loss={test_loss:.4f}")

# store final results
exp_best = experiment_data["max_n_gram_length"]["SPR_BENCH"]
exp_best["best_n"] = best_n
exp_best["test_acc"] = test_acc
exp_best["test_loss"] = test_loss
exp_best["predictions"] = torch.argmax(test_logits, 1).numpy()
exp_best["ground_truth"] = best_artifact["y_test"]

# --------------- SAVE EVERYTHING --------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
