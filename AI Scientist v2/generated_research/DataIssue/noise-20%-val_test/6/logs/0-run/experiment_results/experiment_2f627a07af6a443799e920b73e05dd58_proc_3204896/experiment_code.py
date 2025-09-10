import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import DatasetDict, load_dataset

# ---------------------------------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------------- n-gram vectoriser ---------------
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


# -------------------- MODEL ------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


# -------------- EXPERIMENT STORE -------------------
experiment_data = {
    "batch_size_tuning": {
        "SPR_BENCH": {
            "batch_sizes": [],
            "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test,
        }
    }
}

# -------------- TRAIN / EVAL FUNCS -----------------
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
            preds = logits.argmax(1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            all_logits.append(logits.cpu())
    return correct / total, loss_sum / total, torch.cat(all_logits)


def run_one_experiment(batch_size, epochs=10, top_k=10, lr=1e-3):
    train_loader = DataLoader(
        NgramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=batch_size)
    model = LogReg(num_feats, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
        train_acc, train_loss = correct / total, loss_sum / total
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
        rule_fid = (lin_full.argmax(1) == lin_trunc.argmax(1)).float().mean().item()
        # store epoch metrics
        experiment_data["batch_size_tuning"]["SPR_BENCH"]["metrics"][
            "train_acc"
        ].append(train_acc)
        experiment_data["batch_size_tuning"]["SPR_BENCH"]["metrics"]["val_acc"].append(
            val_acc
        )
        experiment_data["batch_size_tuning"]["SPR_BENCH"]["metrics"][
            "rule_fidelity"
        ].append(rule_fid)
        experiment_data["batch_size_tuning"]["SPR_BENCH"]["losses"]["train"].append(
            train_loss
        )
        experiment_data["batch_size_tuning"]["SPR_BENCH"]["losses"]["val"].append(
            val_loss
        )
        print(
            f"[bs={batch_size}] Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
        )
    # final test evaluation
    test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=batch_size)
    test_acc, test_loss, test_logits = evaluate(model, test_loader)
    print(f"[bs={batch_size}] Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")
    experiment_data["batch_size_tuning"]["SPR_BENCH"]["predictions"].append(
        test_logits.argmax(1).numpy()
    )
    experiment_data["batch_size_tuning"]["SPR_BENCH"]["batch_sizes"].append(batch_size)


# -------------- HYPERPARAM SWEEP -------------------
for bs in [32, 64, 128, 256, 512]:
    run_one_experiment(bs)

# ---------------- SAVE RESULTS ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
