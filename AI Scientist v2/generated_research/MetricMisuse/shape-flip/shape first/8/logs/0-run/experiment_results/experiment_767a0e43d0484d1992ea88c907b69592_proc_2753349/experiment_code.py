# hidden_size_tuning.py
# Hyper-parameter sweep over the hidden layer size of an MLP classifier
import os, pathlib, random, string, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ----------------- reproducibility -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- helper symbolic functions -----------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def rule_signature(sequence: str):
    return (count_shape_variety(sequence), count_color_variety(sequence))


# ----------------- fallback synthetic data -----------------
def random_token():
    shape = random.choice(string.ascii_uppercase[:10])  # 10 shapes
    colour = random.choice(string.digits[:5])  # 5 colours
    return shape + colour


def generate_synthetic_split(n_rows: int, seed=0):
    rnd = random.Random(seed)
    seqs, labels = [], []
    for _ in range(n_rows):
        length = rnd.randint(3, 10)
        seq = " ".join(random_token() for _ in range(length))
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(lbl)
    return {"id": list(range(n_rows)), "sequence": seqs, "label": labels}


def load_spr_bench(root_path: pathlib.Path) -> DatasetDict:
    if root_path.exists():
        print(f"Loading real SPR_BENCH from {root_path}")

        def _load(f):
            return load_dataset("csv", data_files=str(root_path / f), split="train")

        return DatasetDict(
            train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
        )
    print("SPR_BENCH not found – generating synthetic data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, seed=1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, seed=2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, seed=3)),
    )


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

# ----------------- feature encoding -----------------
shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3  # shape histogram + colour histogram + 3 global features


def encode_sequence(seq: str) -> np.ndarray:
    vec = np.zeros(feature_dim, dtype=np.float32)
    toks = seq.split()
    for tok in toks:
        if len(tok) < 2:
            continue
        vec[shape_to_idx[tok[0]]] += 1
        vec[26 + colour_to_idx[tok[1]]] += 1
    vec[-3] = len(toks)
    vec[-2] = count_shape_variety(seq)
    vec[-1] = count_color_variety(seq)
    return vec


def encode_dataset(hf_ds):
    feats = np.stack([encode_sequence(s) for s in hf_ds["sequence"]])
    labels = np.array(hf_ds["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in hf_ds["sequence"]]
    return feats, labels, sigs


# ----------------- torch dataset -----------------
class SPRTorchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


# ----------------- evaluation helper -----------------
def evaluate(model, loader, sigs_all, unseen_sigs):
    model.eval()
    correct = total = correct_unseen = total_unseen = 0
    preds_all = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            preds = logits.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            total += y.size(0)
            correct += (preds == y).sum().item()
            for p, y_true in zip(preds.cpu().numpy(), y.cpu().numpy()):
                sig = sigs_all[idx]
                if sig in unseen_sigs:
                    total_unseen += 1
                    correct_unseen += int(p == y_true)
                idx += 1
    acc = correct / total
    ura = correct_unseen / total_unseen if total_unseen else 0.0
    return acc, ura, preds_all


# ----------------- experiment bookkeeping -----------------
experiment_data = {
    "hidden_size": {
        "SPR_BENCH": {
            "hidden_sizes": [],
            "metrics": {
                "train_acc": [],
                "val_acc": [],
                "test_acc": [],
                "val_ura": [],
                "test_ura": [],
            },
            "losses": {"train": [], "val": []},
            "predictions": [],  # test predictions for the best size
            "ground_truth": [],  # test ground truth (once)
            "timestamps": [],
        }
    }
}

# keep ground truth once
_, y_test_np, _ = encode_dataset(dsets["test"])
experiment_data["hidden_size"]["SPR_BENCH"]["ground_truth"] = y_test_np.tolist()

# ----------------- hyper-parameter sweep -----------------
hidden_sizes = [32, 64, 128, 256]
EPOCHS = 5
best_val_acc = -1
best_preds = []
best_size = None

for hid in hidden_sizes:
    print(f"\n--- Training with hidden size = {hid} ---")
    # data loaders freshly encoded each sweep (cheap)
    X_train, y_train, sig_train = encode_dataset(dsets["train"])
    X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
    X_test, y_test, sig_test = encode_dataset(dsets["test"])
    train_loader = DataLoader(SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True)
    dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
    test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)

    train_sigs_set = set(sig_train)
    unseen_dev_sigs = {s for s in sig_dev if s not in train_sigs_set}
    unseen_test_sigs = {s for s in sig_test if s not in train_sigs_set}

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(feature_dim, hid, len(set(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = running_correct = total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(1)
            running_correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = running_correct / total

    # one final dev evaluation
    val_acc, val_ura, _ = evaluate(model, dev_loader, sig_dev, unseen_dev_sigs)
    train_loss, _ = train_loss, train_acc  # already computed

    # store results
    ed = experiment_data["hidden_size"]["SPR_BENCH"]
    ed["hidden_sizes"].append(hid)
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["val_ura"].append(val_ura)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(0.0)  # placeholder (not tracked every epoch)
    ed["timestamps"].append(time.time())

    print(
        f"Hidden {hid}: train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  URA={val_ura:.3f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_size = hid
        # evaluate on test with the current best
        test_acc, test_ura, test_preds = evaluate(
            model, test_loader, sig_test, unseen_test_sigs
        )
        best_preds = test_preds
        ed_best = (test_acc, test_ura)

# log best test metrics
ed = experiment_data["hidden_size"]["SPR_BENCH"]
ed["metrics"]["test_acc"].append(ed_best[0])
ed["metrics"]["test_ura"].append(ed_best[1])
ed["predictions"] = best_preds
print(
    f"\nBest hidden size = {best_size}  Test Acc = {ed_best[0]:.3f}  Test URA = {ed_best[1]:.3f}"
)

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
