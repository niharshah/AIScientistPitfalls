import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# ----------------- EXPERIMENT STORE -----------------
experiment_data = {
    "weight_decay": {"SPR_BENCH": {}}  # will hold a dict per decay value
}

# ----------------- DEVICE & WORKDIR -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- DATA ------------------------
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


# --------------- N-GRAM VECTORIZER ------------------
def build_vocab(seqs: List[str]):
    unis, bis = set(), set()
    for s in seqs:
        chars = list(s)
        unis.update(chars)
        bis.update([s[i : i + 2] for i in range(len(s) - 1)])
    vocab = sorted(list(unis)) + sorted(list(bis))
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
test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=batch_size)


# -------------------- MODEL ------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            _, preds = torch.max(logits, 1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            all_logits.append(logits.cpu())
    return correct / total, loss_sum / total, torch.cat(all_logits)


# ----------------- HYPERPARAM SEARCH ---------------
decay_values = [0.0, 1e-5, 1e-4, 1e-3]
epochs, top_k = 10, 10

for wd in decay_values:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = LogReg(num_feats, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    # containers for this run
    run_store = {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test,
        "test_acc": None,
        "test_loss": None,
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for batch in train_loader:
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

        val_acc, val_loss, val_logits = evaluate(model, dev_loader)

        # ------- Rule Fidelity -------
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

        run_store["metrics"]["train_acc"].append(train_acc)
        run_store["metrics"]["val_acc"].append(val_acc)
        run_store["metrics"]["rule_fidelity"].append(rule_fid)
        run_store["losses"]["train"].append(train_loss)
        run_store["losses"]["val"].append(val_loss)

        print(
            f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
        )

    # ---------- Final Test ----------
    test_acc, test_loss, test_logits = evaluate(model, test_loader)
    run_store["predictions"] = torch.argmax(test_logits, 1).numpy()
    run_store["test_acc"] = test_acc
    run_store["test_loss"] = test_loss
    print(f"Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")

    experiment_data["weight_decay"]["SPR_BENCH"][str(wd)] = run_store

# ----------------- SAVE RESULTS --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
