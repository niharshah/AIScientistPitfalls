import os, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict
from typing import Dict

# ----------------- I/O & container ----------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"weight_decay": {}}  # mandatory name & file
save_path = os.path.join(working_dir, "experiment_data.npy")

# ----------------- Globals ----------------- #
BATCH_SIZE, VAL_BATCH, LR, EPOCHS, RULE_TOP_K = 256, 512, 1e-2, 10, 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------- Dataset load ----------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    dd = DatasetDict()
    for sp in ["train", "dev", "test"]:
        dd[sp] = _load(f"{sp}.csv")
    return dd


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr.keys())

# ----------------- Vocabulary + vectoriser ----------------- #
char2idx, all_chars = {}, set()
for seq in spr["train"]["sequence"]:
    all_chars.update(seq)
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
vocab_size = len(char2idx)
print("Vocab size:", vocab_size)


def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(vocab_size, np.float32)
    for ch in seq:
        v[char2idx[ch]] += 1.0
    return v / len(seq) if seq else v


def prep(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]])
    y = np.array(split["label"], np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


X_train, y_train = prep(spr["train"])
X_dev, y_dev = prep(spr["dev"])
X_test, y_test = prep(spr["test"])
num_classes = int(max(y_train.max(), y_dev.max(), y_test.max()) + 1)
print("Classes:", num_classes)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=VAL_BATCH)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=VAL_BATCH)


# ----------------- Helpers ----------------- #
class CharBagLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot = cor = loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb)
            loss = criterion(logit, yb)
            loss_sum += loss.item() * yb.size(0)
            pred = logit.argmax(1)
            cor += (pred == yb).sum().item()
            tot += yb.size(0)
    return cor / tot, loss_sum / tot


def rule_accuracy(model, loader):
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]
    tot = cor = 0
    for xb, yb in loader:
        counts = (xb.numpy() * 1000).astype(int)
        preds = []
        for vec in counts:
            votes = [vec[top_idx[c]].sum() for c in range(num_classes)]
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        cor += (preds == yb).sum().item()
        tot += yb.size(0)
    return cor / tot


# ----------------- Hyper-parameter sweep ----------------- #
SEARCH_SPACE = [0.0, 1e-5, 1e-4, 1e-3]

for wd in SEARCH_SPACE:
    print(f"\n==== Training with weight_decay={wd} ====")
    model = CharBagLinear(vocab_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    log = {
        "metrics": {"train_acc": [], "val_acc": [], "RBA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = run_cor = seen = 0
        start = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * yb.size(0)
            run_cor += (out.argmax(1) == yb).sum().item()
            seen += yb.size(0)
        tr_acc, tr_loss = run_cor / seen, run_loss / seen
        val_acc, val_loss = evaluate(model, val_loader)
        rba = rule_accuracy(model, val_loader)
        log["metrics"]["train_acc"].append(tr_acc)
        log["metrics"]["val_acc"].append(val_acc)
        log["metrics"]["RBA"].append(rba)
        log["losses"]["train"].append(tr_loss)
        log["losses"]["val"].append(val_loss)
        log["timestamps"].append(time.time())
        print(
            f"Ep{epoch:02d}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | RBA={rba:.3f} "
            f"time={time.time()-start:.1f}s"
        )

    # final test
    test_acc, test_loss = evaluate(model, test_loader)
    rba_test = rule_accuracy(model, test_loader)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.3f} RBA={rba_test:.3f}")

    # store predictions
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds_all.append(logits.argmax(1).cpu())
            gts_all.append(yb)
    log["predictions"] = torch.cat(preds_all).numpy()
    log["ground_truth"] = torch.cat(gts_all).numpy()
    log["test_metrics"] = {"acc": test_acc, "loss": test_loss, "RBA": rba_test}

    experiment_data["weight_decay"][str(wd)] = {"SPR_BENCH": log}

# ----------------- Save ----------------- #
np.save(save_path, experiment_data)
print(f"\nSaved experiment data to {save_path}")
