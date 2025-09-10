import os, pathlib, random, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- reproducibility --------------------------------
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ------------------- data loader ------------------------------------
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

# ------------------- hashed n-gram vectoriser -----------------------
MAX_N = 4  # up to 4-grams
FEAT_DIM = 16384  # power of two keeps hash(x)&(dim-1) uniform


def ngram_hashes(seq: str):
    """Return list of indices for all 1-4 gram hashes in the sequence."""
    idxs = []
    L = len(seq)
    for n in range(1, MAX_N + 1):
        if L < n:
            break
        for i in range(L - n + 1):
            gram = seq[i : i + n]
            h = hash(gram) & (FEAT_DIM - 1)  # fast power-of-two mod
            idxs.append(h)
    return idxs


class SPRHashedDataset(Dataset):
    def __init__(self, hf_split, label2id):
        self.sp = hf_split
        self.label2id = label2id

    def __len__(self):
        return len(self.sp)

    def __getitem__(self, idx):
        seq = self.sp[idx]["sequence"]
        ys = self.label2id[self.sp[idx]["label"]]
        vec = torch.zeros(FEAT_DIM, dtype=torch.float32)
        for h in ngram_hashes(seq):
            vec[h] += 1.0
        return {"x": vec, "y": torch.tensor(ys)}


labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print("classes:", labels)

batch_size = 128
train_loader = DataLoader(
    SPRHashedDataset(dsets["train"], label2id),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
dev_loader = DataLoader(
    SPRHashedDataset(dsets["dev"], label2id), batch_size=batch_size, num_workers=2
)
test_loader = DataLoader(
    SPRHashedDataset(dsets["test"], label2id), batch_size=batch_size, num_workers=2
)


# ------------------- model ------------------------------------------
class HashedLogReg(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


model = HashedLogReg(FEAT_DIM, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------- experiment data container ----------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------- helper functions -------------------------------
@torch.no_grad()
def evaluate(loader, use_trunc=False, W_trunc=None, b=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds_all = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_trunc:
            # manual forward with truncated weight on CPU (weights small)
            x = batch["x"]
            logits = torch.matmul(x, W_trunc.T) + b
        else:
            logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        pred = logits.argmax(1)
        preds_all.append(pred.cpu())
        total += batch["y"].size(0)
        correct += (pred == batch["y"]).sum().item()
        loss_sum += loss.item() * batch["y"].size(0)
    acc = correct / total
    return acc, loss_sum / total, torch.cat(preds_all)


# ------------------- training loop ----------------------------------
EPOCHS = 8
TOP_K = 15  # rules per class
best_val, best_preds = 0.0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    seen, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        seen += batch["y"].size(0)
        loss_sum += loss.item() * batch["y"].size(0)
        correct += (logits.argmax(1) == batch["y"]).sum().item()
    train_acc = correct / seen
    train_loss = loss_sum / seen

    val_acc, val_loss, _ = evaluate(dev_loader)
    # ------------- rule fidelity ---------------
    W = model.linear.weight.detach().cpu()
    b = model.linear.bias.detach().cpu()
    W_trunc = torch.zeros_like(W)
    for c in range(num_classes):
        top_idx = torch.topk(W[c].abs(), TOP_K).indices
        W_trunc[c, top_idx] = W[c, top_idx]
    rf_acc, _, _ = evaluate(dev_loader, use_trunc=True, W_trunc=W_trunc, b=b)

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["rule_fidelity"].append(rf_acc)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    print(
        f"Epoch {epoch}: "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
        f"RFA={rf_acc:.3f}  ({time.time()-t0:.1f}s)"
    )

    if val_acc > best_val:
        best_val = val_acc
        best_preds = evaluate(test_loader)[2]
        print(f"  new best dev accuracy, stored test predictions.")

# --------------- save experiment ------------------------------------
experiment_data["SPR_BENCH"]["predictions"] = best_preds.numpy().tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = [
    label2id[l] for l in dsets["test"]["label"]
]

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All metrics saved to", os.path.join(working_dir, "experiment_data.npy"))
