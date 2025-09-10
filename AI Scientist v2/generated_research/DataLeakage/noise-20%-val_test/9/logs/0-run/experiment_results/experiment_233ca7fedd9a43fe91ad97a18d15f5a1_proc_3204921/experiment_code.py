import os, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict
from typing import Dict

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Fixed hyper-params ---------------- #
BATCH_SIZE, VAL_BATCH, LR, EPOCHS = 256, 512, 1e-2, 10
RULE_TOP_K = 1
LABEL_SMOOTH_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2]


# ---------------- Dataset loading ---------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr_bench = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr_bench.keys())

# ---------------- Vocabulary ---------------- #
all_chars = set(ch for seq in spr_bench["train"]["sequence"] for ch in seq)
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
vocab_size = len(char2idx)
print(f"Vocab size = {vocab_size}")


def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[char2idx[ch]] += 1.0
    if seq:
        vec /= len(seq)
    return vec


def prepare(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]])
    y = np.array(split["label"], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


X_train, y_train = prepare(spr_bench["train"])
X_dev, y_dev = prepare(spr_bench["dev"])
X_test, y_test = prepare(spr_bench["test"])
num_classes = int(max(y_train.max(), y_dev.max(), y_test.max()) + 1)
print(f"Number of classes: {num_classes}")

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=VAL_BATCH)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=VAL_BATCH)


# ---------------- Model ---------------- #
class CharBagLinear(nn.Module):
    def __init__(self, in_dim: int, num_cls: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        return self.linear(x)


# ---------------- Helper functions ---------------- #
def evaluate(model: nn.Module, loader, criterion):
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


def compute_rule_accuracy(model, loader):
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]
    tot, corr = 0, 0
    for xb, yb in loader:
        counts = (xb.numpy() * 1000).astype(int)
        pred = []
        for vec in counts:
            votes = [vec[top_idx[c]].sum() for c in range(num_classes)]
            pred.append(int(np.argmax(votes)))
        pred = torch.tensor(pred)
        corr += (pred == yb).sum().item()
        tot += yb.size(0)
    return corr / tot


# ---------------- Experiment store ---------------- #
experiment_data: Dict = {"LABEL_SMOOTHING": {}}

# ---------------- Tuning loop ---------------- #
for sm in LABEL_SMOOTH_VALUES:
    print(f"\n=== Training with label_smoothing={sm} ===")
    model = CharBagLinear(vocab_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=sm)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    run_data = {
        "metrics": {"train_acc": [], "val_acc": [], "RBA": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, run_corr, seen = 0.0, 0, 0
        start = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            pred = logits.argmax(1)
            run_loss += loss.item() * yb.size(0)
            run_corr += (pred == yb).sum().item()
            seen += yb.size(0)
        train_acc = run_corr / seen
        train_loss = run_loss / seen
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        rba = compute_rule_accuracy(model, val_loader)

        run_data["metrics"]["train_acc"].append(train_acc)
        run_data["metrics"]["val_acc"].append(val_acc)
        run_data["metrics"]["RBA"].append(rba)
        run_data["losses"]["train"].append(train_loss)
        run_data["losses"]["val"].append(val_loss)
        run_data["timestamps"].append(time.time())

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
            f"RBA={rba:.3f} | time={time.time()-start:.1f}s"
        )

    # Test evaluation
    test_acc, test_loss = evaluate(model, test_loader, criterion)
    rba_test = compute_rule_accuracy(model, test_loader)
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.3f}, RBA={rba_test:.3f}")

    # Store predictions
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(1).cpu())
            gts.append(yb)
    run_data["metrics"]["test_acc"] = test_acc
    run_data["losses"]["test"] = test_loss
    run_data["predictions"] = torch.cat(preds).numpy()
    run_data["ground_truth"] = torch.cat(gts).numpy()

    experiment_data["LABEL_SMOOTHING"][f"{sm}"] = run_data

# ---------------- Save everything ---------------- #
os.makedirs("working", exist_ok=True)
np.save("working/experiment_data.npy", experiment_data)
print("\nAll experiment data saved to working/experiment_data.npy")
