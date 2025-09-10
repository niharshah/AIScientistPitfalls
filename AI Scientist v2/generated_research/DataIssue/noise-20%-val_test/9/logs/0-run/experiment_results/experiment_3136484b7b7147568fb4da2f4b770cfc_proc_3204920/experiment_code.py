import os, time, pathlib, numpy as np, torch
from typing import Dict
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ----------------- Saving dict ----------------- #
experiment_data = {"RULE_TOP_K": {}}  # top-level key required by instructions

# ----------------- Device ----------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- Hyper-params ----------------- #
BATCH_SIZE, VAL_BATCH, LR, EPOCHS = 256, 512, 1e-2, 10
TOP_K_VALUES = [1, 3, 5]  # values to tune


# ----------------- Dataset loading ----------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr_bench = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr_bench.keys())

# ----------------- Vocabulary ----------------- #
all_chars = set(ch for seq in spr_bench["train"]["sequence"] for ch in seq)
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx)
print("Vocab size =", vocab_size)


def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[char2idx[ch]] += 1.0
    if len(seq):
        vec /= len(seq)
    return vec


def prepare_split(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]])
    y = np.array(split["label"], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


X_train, y_train = prepare_split(spr_bench["train"])
X_dev, y_dev = prepare_split(spr_bench["dev"])
X_test, y_test = prepare_split(spr_bench["test"])
num_classes = int(max(y_train.max(), y_dev.max(), y_test.max()) + 1)
print("Number of classes:", num_classes)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=VAL_BATCH)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=VAL_BATCH)


# ----------------- Model ----------------- #
class CharBagLinear(nn.Module):
    def __init__(self, in_dim: int, num_cls: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        return self.linear(x)


model = CharBagLinear(vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ----------------- Helper functions ----------------- #
def evaluate(loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            tot += yb.size(0)
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return correct / tot, loss_sum / tot


def compute_rule_accuracy(loader, top_k: int):
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()  # [C, V]
    top_idx = np.argsort(W, axis=1)[:, -top_k:]
    tot, correct = 0, 0
    for xb, yb in loader:
        counts = (xb.numpy() * 1000).astype(int)
        preds = []
        for cnt_vec in counts:
            votes = [cnt_vec[top_idx[cls]].sum() for cls in range(num_classes)]
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        correct += (preds == yb).sum().item()
        tot += yb.size(0)
    return correct / tot


# ----------------- Initialise experiment_data stubs ----------------- #
for k in TOP_K_VALUES:
    exp_key = f"SPR_BENCH_K{k}"
    experiment_data["RULE_TOP_K"][exp_key] = {
        "metrics": {"train_acc": [], "val_acc": [], "RBA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

# ----------------- Training loop ----------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, run_correct, seen = 0.0, 0, 0
    t0 = time.time()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * yb.size(0)
        run_correct += (logits.argmax(1) == yb).sum().item()
        seen += yb.size(0)
    train_acc, train_loss = run_correct / seen, run_loss / seen
    val_acc, val_loss = evaluate(val_loader)

    # RBA for each K
    rba_vals: Dict[int, float] = {
        k: compute_rule_accuracy(val_loader, k) for k in TOP_K_VALUES
    }

    # store
    for k in TOP_K_VALUES:
        key = f"SPR_BENCH_K{k}"
        ed = experiment_data["RULE_TOP_K"][key]
        ed["metrics"]["train_acc"].append(train_acc)
        ed["metrics"]["val_acc"].append(val_acc)
        ed["metrics"]["RBA"].append(rba_vals[k])
        ed["losses"]["train"].append(train_loss)
        ed["losses"]["val"].append(val_loss)
        ed["timestamps"].append(time.time())

    rba_str = " | ".join([f"K{k}:{rba_vals[k]:.3f}" for k in TOP_K_VALUES])
    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
        f"RBA[{rba_str}] | "
        f"time={time.time()-t0:.1f}s"
    )

# ----------------- Final test evaluation per K ----------------- #
test_acc, test_loss = evaluate(test_loader)
print(f"\nTEST set: loss={test_loss:.4f}, acc={test_acc:.3f}")

model.eval()
all_preds, all_gts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        all_preds.append(logits.argmax(1).cpu())
        all_gts.append(yb)
all_preds = torch.cat(all_preds).numpy()
all_gts = torch.cat(all_gts).numpy()

for k in TOP_K_VALUES:
    key = f"SPR_BENCH_K{k}"
    ed = experiment_data["RULE_TOP_K"][key]
    rba_test = compute_rule_accuracy(test_loader, k)
    ed["metrics"]["test_acc"] = test_acc
    ed["metrics"]["test_RBA"] = rba_test
    ed["losses"]["test"] = test_loss
    ed["predictions"] = all_preds
    ed["ground_truth"] = all_gts
    print(f"K={k:2d} -> Test RBA={rba_test:.3f}")

# ----------------- Save everything ----------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nExperiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
