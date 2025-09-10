import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------------- #
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------- #
# Hyper-parameters
BATCH = 256
VAL_BATCH = 512
EPOCHS = 12
LR = 1e-2
L1_LAMBDA = 1e-3  # strength of sparsity penalty
TOP_K_RULE = 1  # number of chars per class kept in rule


# --------------------------------------------------------------------------- #
# Load SPR_BENCH
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ("train", "dev", "test"):
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
ds = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in ds.items()})

# --------------------------------------------------------------------------- #
# Char vocabulary + bag-of-chars vectors
vocab = sorted({ch for seq in ds["train"]["sequence"] for ch in seq})
char2idx = {c: i for i, c in enumerate(vocab)}
V = len(vocab)


def seq2vec(s: str) -> np.ndarray:
    v = np.zeros(V, np.float32)
    for ch in s:
        v[char2idx[ch]] += 1
    if len(s):
        v /= len(s)
    return v


def split_to_tensors(split):
    X = np.stack([seq2vec(s) for s in split["sequence"]])
    y = np.array(split["label"], np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


Xtr, ytr = split_to_tensors(ds["train"])
Xdv, ydv = split_to_tensors(ds["dev"])
Xte, yte = split_to_tensors(ds["test"])
C = int(max(ytr.max(), ydv.max(), yte.max()) + 1)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(TensorDataset(Xdv, ydv), batch_size=VAL_BATCH)
test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=VAL_BATCH)


# --------------------------------------------------------------------------- #
# Sparse linear classifier
class SparseLinear(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        self.W = nn.Linear(in_dim, num_cls, bias=True)

    def forward(self, x):
        return self.W(x)


model = SparseLinear(V, C).to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------------------------------------------------------------- #
# Metrics container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
    }
}


# --------------------------------------------------------------------------- #
# Helper: evaluation + rule fidelity ---------------------------------------- #
def evaluate(loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            pred = logits.argmax(1)
            tot += yb.size(0)
            correct += (pred == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return correct / tot, loss_sum / tot


def extract_rules():  # returns list[List[int]] indices per class
    with torch.no_grad():
        W = model.W.weight.detach().cpu().numpy()  # [C,V]
    topk = np.argsort(W, axis=1)[:, -TOP_K_RULE:]
    return topk


def rule_predict(x_batch, rules):  # x_batch: [B,V] numpy counts
    votes = np.zeros((x_batch.shape[0], C), np.float32)
    for cls in range(C):
        votes[:, cls] = x_batch[:, rules[cls]].sum(axis=1)
    return votes.argmax(1)


def compute_rule_fidelity(loader):
    rules = extract_rules()
    model.eval()
    agree, total = 0, 0
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            model_preds = logits.argmax(1).cpu().numpy()
            counts = (xb.numpy() * 1000).astype(int)  # rescale to counts
            rule_preds = rule_predict(counts, rules)
            agree += (rule_preds == model_preds).sum()
            total += xb.shape[0]
    return agree / total


# --------------------------------------------------------------------------- #
# Training loop ------------------------------------------------------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    start = time.time()
    seen, correct, loss_accum = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss_ce = criterion(logits, yb)
        l1_pen = sum(p.abs().sum() for p in model.parameters())
        loss = loss_ce + L1_LAMBDA * l1_pen
        loss.backward()
        opt.step()
        pred = logits.argmax(1)
        seen += yb.size(0)
        correct += (pred == yb).sum().item()
        loss_accum += loss_ce.item() * yb.size(0)  # record pure CE loss
    train_acc = correct / seen
    train_loss = loss_accum / seen
    val_acc, val_loss = evaluate(val_loader)
    fidelity = compute_rule_fidelity(val_loader)

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["rule_fidelity"].append(fidelity)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: "
        f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
        f"fidelity={fidelity:.3f}, val_loss={val_loss:.4f}, "
        f"time={time.time()-start:.1f}s"
    )

# --------------------------------------------------------------------------- #
# Final evaluation ---------------------------------------------------------- #
test_acc, test_loss = evaluate(test_loader)
test_fidelity = compute_rule_fidelity(test_loader)
print(
    f"\nTEST  – acc: {test_acc:.3f}, loss: {test_loss:.4f}, "
    f"Rule Fidelity: {test_fidelity:.3f}"
)

# --------------------------------------------------------------------------- #
# Save everything ----------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved metrics -> {os.path.join(working_dir, 'experiment_data.npy')}")
