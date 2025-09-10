import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import DatasetDict, load_dataset

# ------------------ SETTINGS -----------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
batch_size, epochs, top_k = 128, 10, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------- DATA LOADING ----------------------
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


DATA_PATH = pathlib.Path(
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)  # adjust if needed
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ------------- N-GRAM FEATURES --------------------
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
            v[idx[c]] += 1
    for i in range(len(seq) - 1):
        bg = seq[i : i + 2]
        if bg in idx:
            v[idx[bg]] += 1
    return v


vocab_idx = build_vocab(dsets["train"]["sequence"])
num_feats = len(vocab_idx)
print("Feature size:", num_feats)

labels = sorted(set(dsets["train"]["label"]))
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


train_loader = DataLoader(
    NgramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=batch_size)

# ---------------- EXPERIMENT STORE ----------------
experiment_data = {"weight_decay": {}}


# ------------------ MODEL DEF ---------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()


# -------------- TRAINING / EVAL FUNCS -------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_logits = []
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


# ================== HYPERPARAM SWEEP ==============
for wd in weight_decays:
    print(f"\n===== Training with weight_decay={wd} =====")
    torch.manual_seed(0)
    model = LogReg(num_feats, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    run_data = {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
        "predictions": None,
        "ground_truth": y_test,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
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
        val_acc, val_loss, _ = evaluate(model, dev_loader)

        # ----- Rule fidelity -----
        with torch.no_grad():
            W = model.linear.weight.cpu().numpy()
            b = model.linear.bias.cpu().numpy()
            W_trunc = np.zeros_like(W)
            for c in range(num_classes):
                idxs = np.argsort(-np.abs(W[c]))[:top_k]
                W_trunc[c, idxs] = W[c, idxs]
            lin_full = torch.from_numpy((X_dev @ W.T) + b)
            lin_trunc = torch.from_numpy((X_dev @ W_trunc.T) + b)
            rule_fid = (lin_trunc.argmax(1) == lin_full.argmax(1)).float().mean().item()

        run_data["metrics"]["train_acc"].append(train_acc)
        run_data["metrics"]["val_acc"].append(val_acc)
        run_data["metrics"]["rule_fidelity"].append(rule_fid)
        run_data["losses"]["train"].append(train_loss)
        run_data["losses"]["val"].append(val_loss)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
        )

    # -------- Final test evaluation -------------
    test_acc, test_loss, test_logits = evaluate(model, test_loader)
    run_data["test_acc"], run_data["test_loss"] = test_acc, test_loss
    run_data["predictions"] = test_logits.argmax(1).numpy()
    print(f"Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")

    experiment_data["weight_decay"][str(wd)] = run_data

# -------------- SAVE EVERYTHING ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
