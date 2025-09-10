import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------- WORK DIR & DEVICE -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------- REPRODUCIBILITY ---------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ------------------------- DATA LOADING ------------------------------
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


# ------------------------- VOCAB BUILDING ----------------------------
def build_vocab(seqs, max_n=3):
    vocab = set()
    for s in seqs:
        for n in range(1, max_n + 1):
            vocab.update([s[i : i + n] for i in range(len(s) - n + 1)])
    return {tok: i for i, tok in enumerate(sorted(vocab))}


vocab_idx = build_vocab(dsets["train"]["sequence"], max_n=3)
print(f"Vocabulary size (1-3 grams): {len(vocab_idx)}")

labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_feats, num_classes = len(vocab_idx), len(labels)


def vectorise(seq, idx, max_n=3):
    v = np.zeros(len(idx), dtype=np.float32)
    for n in range(1, max_n + 1):
        for i in range(len(seq) - n + 1):
            tok = seq[i : i + n]
            if tok in idx:
                v[idx[tok]] += 1.0
    return v


def encode_split(split):
    X = np.stack([vectorise(s, vocab_idx) for s in dsets[split]["sequence"]])
    y = np.array([label2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y


X_train, y_train = encode_split("train")
X_dev, y_dev = encode_split("dev")
X_test, y_test = encode_split("test")


class GramDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X.astype(np.float32), y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.X[idx]), "y": torch.tensor(self.y[idx])}


batch_size = 128
train_loader = DataLoader(
    GramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(GramDataset(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(GramDataset(X_test, y_test), batch_size=batch_size)


# ------------------------- MODEL -------------------------------------
class SparseLogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()

# ------------------- EXPERIMENT DATA STRUCTURE -----------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "RFA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "lambdas": [],
        "best_lambda": None,
    }
}


# ------------------------- TRAIN / EVAL HELPERS ----------------------
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    logits_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            preds = logits.argmax(1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            logits_all.append(logits.cpu())
    return correct / total, loss_sum / total, torch.cat(logits_all)


def rule_fidelity(model, X, top_k=10):
    W = model.linear.weight.detach().cpu().numpy()
    b = model.linear.bias.detach().cpu().numpy()
    W_top = np.zeros_like(W)
    for c in range(W.shape[0]):
        idxs = np.argsort(-np.abs(W[c]))[:top_k]
        W_top[c, idxs] = W[c, idxs]
    full = torch.from_numpy(X @ W.T + b)
    trunc = torch.from_numpy(X @ W_top.T + b)
    return (torch.argmax(full, 1) == torch.argmax(trunc, 1)).float().mean().item()


# ------------------------- TRAINING GRID -----------------------------
lambdas = [0.0, 1e-3, 1e-2, 1e-1]
epochs = 10
best_dev_acc, best_state, best_lambda = -1.0, None, None

for lam in lambdas:
    print(f"\n===== Training with L1 λ={lam} =====")
    experiment_data["SPR_BENCH"]["lambdas"].append(lam)
    model = SparseLogReg(num_feats, num_classes).to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    hist_train_acc, hist_val_acc, hist_rfa = [], [], []
    hist_train_loss, hist_val_loss = [], []

    for ep in range(1, epochs + 1):
        model.train()
        seen, correct, run_loss = 0, 0, 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimiser.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            if lam > 0:
                l1 = lam * torch.norm(model.linear.weight, p=1)
                loss = loss + l1
            loss.backward()
            optimiser.step()
            run_loss += loss.item() * batch["y"].size(0)
            correct += (logits.argmax(1) == batch["y"]).sum().item()
            seen += batch["y"].size(0)

        train_loss = run_loss / seen
        train_acc = correct / seen
        val_acc, val_loss, _ = evaluate(model, dev_loader)
        rfa = rule_fidelity(model, X_dev, top_k=10)

        print(f"Epoch {ep}: validation_loss = {val_loss:.4f}")
        print(
            f"        train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} RFA={rfa:.3f}"
        )

        hist_train_acc.append(train_acc)
        hist_val_acc.append(val_acc)
        hist_rfa.append(rfa)
        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(hist_train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(hist_val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["RFA"].append(hist_rfa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(hist_train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(hist_val_loss)

    if hist_val_acc[-1] > best_dev_acc:
        best_dev_acc = hist_val_acc[-1]
        best_lambda = lam
        best_state = model.state_dict()

# ------------------------- TEST EVALUATION ---------------------------
print(f"\nBest λ on dev: {best_lambda} (dev_acc={best_dev_acc:.3f})")
best_model = SparseLogReg(num_feats, num_classes).to(device)
best_model.load_state_dict(best_state)
test_acc, test_loss, test_logits = evaluate(best_model, test_loader)
print(f"Test accuracy = {test_acc:.3f}")

experiment_data["SPR_BENCH"]["best_lambda"] = best_lambda
experiment_data["SPR_BENCH"]["predictions"] = test_logits.argmax(1).tolist()


# convert lists to numpy arrays for saving
def to_np_list(lst):
    return np.array(lst, dtype=object)


ed = experiment_data["SPR_BENCH"]
ed["metrics"]["train_acc"] = to_np_list(ed["metrics"]["train_acc"])
ed["metrics"]["val_acc"] = to_np_list(ed["metrics"]["val_acc"])
ed["metrics"]["RFA"] = to_np_list(ed["metrics"]["RFA"])
ed["losses"]["train"] = to_np_list(ed["losses"]["train"])
ed["losses"]["val"] = to_np_list(ed["losses"]["val"])
ed["predictions"] = np.array(ed["predictions"])
ed["ground_truth"] = np.array(ed["ground_truth"])
ed["lambdas"] = np.array(ed["lambdas"])

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all metrics to", os.path.join(working_dir, "experiment_data.npy"))
