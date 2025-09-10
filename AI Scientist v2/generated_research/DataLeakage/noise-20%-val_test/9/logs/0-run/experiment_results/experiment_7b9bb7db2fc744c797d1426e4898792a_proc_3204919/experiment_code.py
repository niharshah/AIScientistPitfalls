import os, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- House-keeping ---------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# ---------------- Hyper-params ---------------- #
BATCH_SIZE, VAL_BATCH, EPOCHS, RULE_TOP_K = 256, 512, 10, 1
LR_CANDIDATES = [1e-2, 3e-3, 1e-3, 3e-4]  # ← search space
STEP_GAMMA, STEP_SIZE = 0.5, 5  # lr decay schedule


# ---------------- Dataset loading ---------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for spl in ["train", "dev", "test"]:
        out[spl] = _load(f"{spl}.csv")
    return out


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr_bench = load_spr_bench(DATA_PATH)

# ---------------- Vocabulary ---------------- #
all_chars = {ch for seq in spr_bench["train"]["sequence"] for ch in seq}
char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
vocab_size = len(char2idx)


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

# ---------------- Dataloaders ---------------- #
train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=VAL_BATCH)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=VAL_BATCH)


# ---------------- Model definition ---------------- #
class CharBagLinear(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        return self.linear(x)


# ---------------- Helpers ---------------- #
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total = correct = loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return correct / total, loss_sum / total


def compute_rule_accuracy(model, loader):
    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy()
    top_idx = np.argsort(W, axis=1)[:, -RULE_TOP_K:]
    total = correct = 0
    for xb, yb in loader:
        cnts = (xb.numpy() * 1000).astype(int)
        preds = []
        for vec in cnts:
            votes = [vec[top_idx[c]].sum() for c in range(num_classes)]
            preds.append(int(np.argmax(votes)))
        preds = torch.tensor(preds)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total


# ---------------- Experiment store ---------------- #
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}

# ---------------- Hyper-parameter loop ---------------- #
for lr in LR_CANDIDATES:
    run_key = f"{lr:.0e}" if lr < 1 else str(lr)
    print(f"\n=== Training with learning rate {lr} ===")
    model = CharBagLinear(vocab_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA
    )
    store = {
        "metrics": {"train_acc": [], "val_acc": [], "RBA": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = run_corr = seen = 0
        start_t = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            run_loss += loss.item() * yb.size(0)
            run_corr += (preds == yb).sum().item()
            seen += yb.size(0)
        scheduler.step()

        train_acc = run_corr / seen
        train_loss = run_loss / seen
        val_acc, val_loss = evaluate(model, val_loader)
        rba = compute_rule_accuracy(model, val_loader)
        store["metrics"]["train_acc"].append(train_acc)
        store["metrics"]["val_acc"].append(val_acc)
        store["metrics"]["RBA"].append(rba)
        store["losses"]["train"].append(train_loss)
        store["losses"]["val"].append(val_loss)
        store["timestamps"].append(time.time())
        print(
            f"Epoch {epoch:02d} | lr={lr:.0e} | "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} RBA={rba:.3f}"
        )

    # -------- Test evaluation -------- #
    test_acc, test_loss = evaluate(model, test_loader)
    rba_test = compute_rule_accuracy(model, test_loader)
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds_all.append(logits.argmax(1).cpu())
            gts_all.append(yb)
    store["test_perf"] = {"acc": test_acc, "loss": test_loss, "RBA": rba_test}
    store["predictions"] = torch.cat(preds_all).numpy()
    store["ground_truth"] = torch.cat(gts_all).numpy()

    experiment_data["learning_rate"]["SPR_BENCH"][run_key] = store
    print(f"Finished LR {lr}: test_acc={test_acc:.3f}, RBA={rba_test:.3f}")

# ---------------- Save everything ---------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
