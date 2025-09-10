import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import DatasetDict, load_dataset

# --------------------------- SETUP ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- DATA ----------------------------------
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


# ------------------- n-gram vectoriser ------------------------------
def build_vocab(seqs: List[str]):
    unis, bis = set(), set()
    for s in seqs:
        unis.update(s)
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
labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Feature size: {num_feats}  #classes: {num_classes}")


def encode_split(split: str):
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

    def __getitem__(self, i):
        return {"x": torch.from_numpy(self.X[i]), "y": torch.tensor(self.y[i])}


# ------------------------- MODEL -----------------------------------
class LogReg(nn.Module):
    def __init__(self, d_in, num_cls):
        super().__init__()
        self.linear = nn.Linear(d_in, num_cls)

    def forward(self, x):
        return self.linear(x)


# -------------------- EXPERIMENT STORE -----------------------------
experiment_data = {"batch_size": {"SPR_BENCH": {}}}

# ----------------------- HYPERPARAM LOOP ---------------------------
batch_sizes = [32, 64, 128, 256]
epochs = 10
top_k = 10

for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    train_loader = DataLoader(
        NgramDataset(X_train, y_train), batch_size=bs, shuffle=True
    )
    dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=bs)
    test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=bs)

    model = LogReg(num_feats, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    run_data = {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test,
    }

    def evaluate(loader):
        model.eval()
        total = correct = loss_sum = 0.0
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

    for epoch in range(1, epochs + 1):
        model.train()
        tot = correct = loss_sum = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            bsz = batch["y"].size(0)
            loss_sum += loss.item() * bsz
            tot += bsz
            correct += (logits.argmax(1) == batch["y"]).sum().item()
        train_acc = correct / tot
        train_loss = loss_sum / tot
        val_acc, val_loss, _ = evaluate(dev_loader)

        # Rule fidelity
        W = model.linear.weight.detach().cpu().numpy()
        b = model.linear.bias.detach().cpu().numpy()
        W_trunc = np.zeros_like(W)
        for c in range(num_classes):
            idx = np.argsort(-np.abs(W[c]))[:top_k]
            W_trunc[c, idx] = W[c, idx]
        lin_full = (X_dev @ W.T) + b
        lin_trunc = (X_dev @ W_trunc.T) + b
        rule_fid = (lin_trunc.argmax(1) == lin_full.argmax(1)).mean()

        run_data["metrics"]["train_acc"].append(train_acc)
        run_data["metrics"]["val_acc"].append(val_acc)
        run_data["metrics"]["rule_fidelity"].append(rule_fid)
        run_data["losses"]["train"].append(train_loss)
        run_data["losses"]["val"].append(val_loss)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
        )

    # test evaluation
    test_acc, test_loss, test_logits = evaluate(test_loader)
    run_data["predictions"] = test_logits.argmax(1).numpy()
    print(f"[bs={bs}] Test_acc={test_acc:.3f} test_loss={test_loss:.4f}")

    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = run_data

# ----------------------- SAVE RESULTS ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
