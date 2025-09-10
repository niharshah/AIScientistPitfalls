import os, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict

# ---------------- I/O & storage ---------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"Adam_Beta1": {"SPR_BENCH": {}}}

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Hyper-params ---------------- #
BATCH_SIZE, VAL_BATCH, LR, EPOCHS = 256, 512, 1e-2, 10
RULE_TOP_K = 1
BETA1_GRID = [0.7, 0.8, 0.9]


# ---------------- Dataset loading ---------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):  # helper
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    ds = DatasetDict()
    for split in ["train", "dev", "test"]:
        ds[split] = _load(f"{split}.csv")
    return ds


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr.keys())

# ---------------- Vocabulary ---------------- #
chars = sorted({c for s in spr["train"]["sequence"] for c in s})
char2idx = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)


def seq_to_vec(seq: str):
    vec = np.zeros(vocab_size, np.float32)
    for ch in seq:
        vec[char2idx[ch]] += 1.0
    if seq:
        vec /= len(seq)
    return vec


def prep(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]])
    y = np.array(split["label"], np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


X_tr, y_tr = prep(spr["train"])
X_val, y_val = prep(spr["dev"])
X_te, y_te = prep(spr["test"])
num_classes = int(max(y_tr.max(), y_val.max(), y_te.max()) + 1)
print("Classes:", num_classes)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), VAL_BATCH)
test_loader = DataLoader(TensorDataset(X_te, y_te), VAL_BATCH)


# ---------------- Model def ---------------- #
class CharBagLinear(nn.Module):
    def __init__(self, dim, cls):
        super().__init__()
        self.linear = nn.Linear(dim, cls)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()


# ---------- Helpers ---------- #
def evaluate(model, loader):
    model.eval()
    tot = cor = loss_sum = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            tot += yb.size(0)
            cor += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return cor / tot, loss_sum / tot


def compute_rba(model, loader):
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]
    tot = cor = 0
    for xb, yb in loader:
        cnts = (xb.numpy() * 1000).astype(int)
        preds = []
        for cvec in cnts:
            votes = [cvec[top_idx[cls]].sum() for cls in range(num_classes)]
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        cor += (preds == yb).sum().item()
        tot += yb.size(0)
    return cor / tot


# ---------------- Hyper-parameter loop ---------------- #
for b1 in BETA1_GRID:
    beta_key = f"beta1={b1}"
    print(f"\n--- Training with {beta_key} ---")
    model = CharBagLinear(vocab_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(b1, 0.999))

    # storage for this run
    run_store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "RBA": [],
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        seen = cor = loss_sum = 0
        start = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            seen += yb.size(0)
            cor += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
        train_acc, train_loss = cor / seen, loss_sum / seen
        val_acc, val_loss = evaluate(model, val_loader)
        rba = compute_rba(model, val_loader)

        run_store["metrics"]["train"].append(train_acc)
        run_store["metrics"]["val"].append(val_acc)
        run_store["losses"]["train"].append(train_loss)
        run_store["losses"]["val"].append(val_loss)
        run_store["RBA"].append(rba)
        print(
            f"Ep{epoch:02d} | tr_acc={train_acc:.3f} val_acc={val_acc:.3f} rba={rba:.3f} "
            f"| {time.time()-start:.1f}s"
        )

    # final test evaluation & predictions
    test_acc, test_loss = evaluate(model, test_loader)
    rba_test = compute_rba(model, test_loader)
    print(f"Test | acc={test_acc:.3f} loss={test_loss:.4f} rba={rba_test:.3f}")
    model.eval()
    preds_all = []
    gts_all = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds_all.append(logits.argmax(1).cpu())
            gts_all.append(yb)
    run_store["predictions"] = torch.cat(preds_all).numpy()
    run_store["ground_truth"] = torch.cat(gts_all).numpy()

    experiment_data["Adam_Beta1"]["SPR_BENCH"][beta_key] = run_store

# ---------------- Save everything ---------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy in", working_dir)
