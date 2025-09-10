# Hyper-parameter tuning: Optimizer choice (Adam vs. SGD+momentum)
import os, time, pathlib, numpy as np, torch
from typing import Dict
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Fixed hyper-params ---------------- #
BATCH_SIZE, VAL_BATCH, EPOCHS = 256, 512, 10
RULE_TOP_K = 1  # characters per class for symbolic rule


# ---------------- Dataset loading ---------------- #
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

# ---------------- Vocabulary ---------------- #
all_chars = set(ch for s in spr_bench["train"]["sequence"] for ch in s)
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
vocab_size = len(char2idx)
print(f"Vocab size = {vocab_size}")


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
print(f"Number of classes: {num_classes}")

# ---------------- DataLoaders ---------------- #
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


# ---------------- Helpers ---------------- #
criterion = nn.CrossEntropyLoss()


def evaluate_model(model: nn.Module, loader) -> Dict[str, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return {"acc": correct / total, "loss": loss_sum / total}


def compute_rule_accuracy(model: nn.Module, loader) -> float:
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()  # [C, V]
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]  # [C, K]
    total, correct = 0, 0
    for xb, yb in loader:
        seq_vecs = xb.numpy()
        counts = (seq_vecs * 1000).astype(int)
        preds = []
        for cnt in counts:
            votes = [cnt[top_idx[cls]].sum() for cls in range(num_classes)]
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
    return correct / total


# ---------------- Hyper-parameter grid ---------------- #
optim_grid = [
    ("Adam", {"lr": 1e-2}),
    ("SGD", {"lr": 0.1, "momentum": 0.9}),
    ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("SGD", {"lr": 0.05, "momentum": 0.8}),
]

# ---------------- Experiment store ---------------- #
experiment_data = {
    "optimizer_choice": {
        "SPR_BENCH": {"configs": []}  # each entry holds data for one hyperparameter run
    }
}

# ---------------- Tuning Loop ---------------- #
for cfg_id, (opt_name, opt_params) in enumerate(optim_grid, 1):
    print(f"\n=== Run {cfg_id}: {opt_name} {opt_params} ===")
    model = CharBagLinear(vocab_size, num_classes).to(device)
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **opt_params)
    else:
        optimizer = torch.optim.SGD(model.parameters(), **opt_params)
    run_data = {
        "optimizer": opt_name,
        "params": opt_params,
        "metrics": {"train_acc": [], "val_acc": [], "RBA": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_t = time.time()
        seen, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            seen += yb.size(0)
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
        train_acc, train_loss = correct / seen, loss_sum / seen

        val_stats = evaluate_model(model, val_loader)
        rba = compute_rule_accuracy(model, val_loader)

        run_data["metrics"]["train_acc"].append(train_acc)
        run_data["metrics"]["val_acc"].append(val_stats["acc"])
        run_data["metrics"]["RBA"].append(rba)
        run_data["losses"]["train"].append(train_loss)
        run_data["losses"]["val"].append(val_stats["loss"])
        run_data["timestamps"].append(time.time())

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['acc']:.3f} | "
            f"RBA={rba:.3f} | "
            f"time={time.time()-start_t:.1f}s"
        )

    # Final test evaluation
    test_stats = evaluate_model(model, test_loader)
    test_rba = compute_rule_accuracy(model, test_loader)
    print(
        f"Test: loss={test_stats['loss']:.4f}, acc={test_stats['acc']:.3f}, RBA={test_rba:.3f}"
    )

    # Store predictions
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds_all.append(logits.argmax(1).cpu())
            gts_all.append(yb)
    run_data["test"] = {
        "loss": test_stats["loss"],
        "acc": test_stats["acc"],
        "RBA": test_rba,
    }
    run_data["predictions"] = torch.cat(preds_all).numpy()
    run_data["ground_truth"] = torch.cat(gts_all).numpy()

    experiment_data["optimizer_choice"]["SPR_BENCH"]["configs"].append(run_data)

# ---------------- Save everything ---------------- #
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
print("\nAll experiment data saved to working/experiment_data.npy")
