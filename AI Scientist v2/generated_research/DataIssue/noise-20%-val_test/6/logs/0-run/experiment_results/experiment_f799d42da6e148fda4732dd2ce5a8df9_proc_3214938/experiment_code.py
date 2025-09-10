import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------- SET-UP
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ------------------------------------------------- DATA
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
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


# build n-gram vocab (uni+bi+tri)
def build_vocab(seqs):
    vocab = set()
    for s in seqs:
        for i in range(len(s)):
            vocab.add(s[i])  # uni
            if i + 1 < len(s):
                vocab.add(s[i : i + 2])  # bi
            if i + 2 < len(s):
                vocab.add(s[i : i + 3])  # tri
    return {tok: i for i, tok in enumerate(sorted(vocab))}


vocab = build_vocab(dsets["train"]["sequence"])
print("Vocab size:", len(vocab))


def vectorise(seq, idx):
    v = np.zeros(len(idx), dtype=np.float32)
    for i in range(len(seq)):
        tok = seq[i]
        v[idx[tok]] += 1
        if i + 1 < len(seq):
            tok = seq[i : i + 2]
            v[idx.get(tok, 0)] += 1 if tok in idx else 0
        if i + 2 < len(seq):
            tok = seq[i : i + 3]
            v[idx.get(tok, 0)] += 1 if tok in idx else 0
    return v


labels = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}


def enc_split(split):
    X = np.stack([vectorise(s, vocab) for s in dsets[split]["sequence"]])
    y = np.array([lab2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y


X_train, y_train = enc_split("train")
X_dev, y_dev = enc_split("dev")
X_test, y_test = enc_split("test")


class GramDS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": torch.from_numpy(self.X[i]), "y": torch.tensor(self.y[i])}


batch_size = 256
train_loader = DataLoader(GramDS(X_train, y_train), batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(GramDS(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(GramDS(X_test, y_test), batch_size=batch_size)


# ------------------------------------------------- MODEL
class SparseLogReg(nn.Module):
    def __init__(self, dim, num_cls):
        super().__init__()
        self.linear = nn.Linear(dim, num_cls, bias=True)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()

# ------------------------------------------------- EXPERIMENT RECORD
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "best_lambda": None,
    }
}


# ------------------------------------------------- HELPERS
def eval_model(model, loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            _, preds = torch.max(logits, 1)
            tot += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            all_logits.append(logits.cpu())
    return correct / tot, loss_sum / tot, torch.cat(all_logits)


def rule_fidelity(model, top_k, X, y):
    W = model.linear.weight.detach().cpu().numpy()
    b = model.linear.bias.detach().cpu().numpy()
    W_trim = np.zeros_like(W)
    for c in range(W.shape[0]):
        idx = np.argsort(-np.abs(W[c]))[:top_k]
        W_trim[c, idx] = W[c, idx]
    logits_full = X @ W.T + b
    logits_rule = X @ W_trim.T + b
    return (np.argmax(logits_full, 1) == np.argmax(logits_rule, 1)).mean()


# ------------------------------------------------- TRAINING
lambdas = [0.0, 1e-4, 5e-4, 1e-3]
epochs = 15
top_k = 10
best_val = -1

for lam in lambdas:
    print(f"\n--- Training λ={lam} ---")
    model = SparseLogReg(len(vocab), len(labels)).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    run_tr_acc, run_val_acc, run_rf = [], [], []
    run_tr_loss, run_val_loss = [], []

    for ep in range(1, epochs + 1):
        model.train()
        seen, correct, tr_loss = 0, 0, 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            l1 = lam * model.linear.weight.abs().mean()
            (loss + l1).backward()
            opt.step()
            tr_loss += loss.item() * batch["y"].size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == batch["y"]).sum().item()
            seen += batch["y"].size(0)
        tr_acc = correct / seen
        tr_loss /= seen

        val_acc, val_loss, _ = eval_model(model, dev_loader)
        rf = rule_fidelity(model, top_k, X_dev, y_dev)

        run_tr_acc.append(tr_acc)
        run_val_acc.append(val_acc)
        run_rf.append(rf)
        run_tr_loss.append(tr_loss)
        run_val_loss.append(val_loss)

        print(f"Epoch {ep}: val_acc={val_acc:.3f} val_loss={val_loss:.4f} RFA={rf:.3f}")

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(run_tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(run_val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["rule_fidelity"].append(run_rf)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(run_tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(run_val_loss)

    if run_val_acc[-1] > best_val:
        best_val = run_val_acc[-1]
        experiment_data["SPR_BENCH"]["best_lambda"] = lam
        test_acc, test_loss, test_logits = eval_model(model, test_loader)
        experiment_data["SPR_BENCH"]["predictions"] = (
            torch.argmax(test_logits, 1).cpu().numpy().tolist()
        )
        print(f"*** New best λ={lam} | Test accuracy={test_acc:.3f} ***")

# ------------------------------------------------- SAVE
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
