import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import DatasetDict, load_dataset

# -------------------- EXPERIMENT STORE --------------------
experiment_data = {
    "adam_beta1": {"SPR_BENCH": {}}  # hyper-param tuning type  # dataset
}

# -------------------- ENV & PATHS -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- DATA ------------------------------
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


# ---------------- n-gram vectoriser -----------------------
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


train_seqs = dsets["train"]["sequence"]
vocab_idx = build_vocab(train_seqs)
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


# -------------------- MODEL -------------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()
epochs = 10
top_k = 10  # n-grams kept per class for rule fidelity
beta1_grid = [0.8, 0.85, 0.9, 0.95]


# ----------------- TRAINING & TUNING ----------------------
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            preds = torch.argmax(logits, 1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            all_logits.append(logits.cpu())
    return correct / total, loss_sum / total, torch.cat(all_logits)


for beta1 in beta1_grid:
    print(f"\n===== Training with beta1 = {beta1} =====")
    model = LogReg(num_feats, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(beta1, 0.999))

    metrics = {"train_acc": [], "val_acc": [], "rule_fidelity": []}
    losses = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        for batch in train_loader_full:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
            preds = torch.argmax(logits, 1)
            correct += (preds == batch["y"]).sum().item()
            seen += batch["y"].size(0)
        train_loss = running_loss / seen
        train_acc = correct / seen

        val_acc, val_loss, _ = evaluate(model, dev_loader)

        # ------ Rule Fidelity ------
        W = model.linear.weight.detach().cpu().numpy()
        b = model.linear.bias.detach().cpu().numpy()
        W_trunc = np.zeros_like(W)
        for c in range(num_classes):
            idxs = np.argsort(-np.abs(W[c]))[:top_k]
            W_trunc[c, idxs] = W[c, idxs]
        lin_trunc = torch.from_numpy((X_dev @ W_trunc.T) + b)
        lin_full = torch.from_numpy((X_dev @ W.T) + b)
        rule_fid = (
            (torch.argmax(lin_trunc, 1) == torch.argmax(lin_full, 1))
            .float()
            .mean()
            .item()
        )

        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        metrics["rule_fidelity"].append(rule_fid)
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
        )

    # -------- Final Test Eval ----------
    test_acc, test_loss, test_logits = evaluate(model, test_loader)

    experiment_data["adam_beta1"]["SPR_BENCH"][str(beta1)] = {
        "metrics": metrics,
        "losses": losses,
        "predictions": torch.argmax(test_logits, 1).numpy(),
        "ground_truth": y_test,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }
    print(f"beta1={beta1} -> Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")

# ---------------- SAVE EXPERIMENT -------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
