import os, pathlib, random, string, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ------------------- housekeeping -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------- reproducibility -------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


# ------------------- helper functions -------------------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def rule_signature(seq: str):
    return (count_shape_variety(seq), count_color_variety(seq))


# ---------- fallback synthetic data ----------
def random_token():
    shape = random.choice(string.ascii_uppercase[:10])  # 10 shapes
    colour = random.choice(string.digits[:5])  # 5 colours
    return shape + colour


def generate_synthetic_split(n_rows: int, seed=0):
    random.seed(seed)
    seqs, labels = [], []
    for _ in range(n_rows):
        length = random.randint(3, 10)
        seq = " ".join(random_token() for _ in range(length))
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(lbl)
    return {"id": list(range(n_rows)), "sequence": seqs, "label": labels}


def load_spr_bench(root_path: pathlib.Path) -> DatasetDict:
    if root_path.exists():
        print(f"Loading real SPR_BENCH from {root_path}")

        def _load(fname):
            return load_dataset("csv", data_files=str(root_path / fname), split="train")

        return DatasetDict(
            train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
        )
    print("SPR_BENCH not found – generating synthetic data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, seed=1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, seed=2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, seed=3)),
    )


# ------------------- load data -------------------
DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3  # shapes hist + colours hist + {len, shapeVar, colourVar}


def encode_sequence(seq: str) -> np.ndarray:
    vec = np.zeros(feature_dim, dtype=np.float32)
    tokens = seq.split()
    for tok in tokens:
        if len(tok) < 2:
            continue
        s, c = tok[0], tok[1]
        vec[shape_to_idx[s]] += 1
        vec[26 + colour_to_idx[c]] += 1
    vec[-3] = len(tokens)
    vec[-2] = count_shape_variety(seq)
    vec[-1] = count_color_variety(seq)
    return vec


def encode_dataset(hf_ds):
    feats = np.stack([encode_sequence(s) for s in hf_ds["sequence"]])
    labels = np.array(hf_ds["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in hf_ds["sequence"]]
    return feats, labels, sigs


# encode once
X_train_all, y_train_all, sig_train = encode_dataset(dsets["train"])
X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
X_test, y_test, sig_test = encode_dataset(dsets["test"])

# unseen signatures for URA
train_sigs_set = set(sig_train)
unseen_dev_sigs = {s for s in sig_dev if s not in train_sigs_set}
unseen_test_sigs = {s for s in sig_test if s not in train_sigs_set}


# ------------------- torch dataset -------------------
class SPRTorchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)


# ------------------- model -------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


criterion = nn.CrossEntropyLoss()


def evaluate(loader, sigs_all, unseen_sigs, model):
    model.eval()
    total = correct = total_unseen = correct_unseen = 0
    all_preds = []
    with torch.no_grad():
        idx = 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            total += y.size(0)
            correct += (preds == y).sum().item()
            for p, y_true in zip(preds.cpu().numpy(), y.cpu().numpy()):
                sig = sigs_all[idx]
                if sig in unseen_sigs:
                    total_unseen += 1
                    if p == y_true:
                        correct_unseen += 1
                idx += 1
    acc = correct / total
    ura = correct_unseen / total_unseen if total_unseen else 0.0
    return acc, ura, all_preds


# ------------------- hyperparameter sweep -------------------
batch_sizes = [16, 32, 64, 128, 256]
EPOCHS = 5

experiment_data = {
    "batch_size_tuning": {
        "SPR_BENCH": {
            "batch_sizes": batch_sizes,
            "metrics": {
                "train_acc": [],
                "val_acc": [],
                "val_ura": [],
                "test_acc": [],
                "test_ura": [],
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "timestamps": [],
        }
    }
}

for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    # fresh model & optimiser
    model = MLP(feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # make loader using the current batch size
    train_loader = DataLoader(
        SPRTorchDS(X_train_all, y_train_all), batch_size=bs, shuffle=True
    )

    epoch_train_acc, epoch_val_acc, epoch_val_ura, epoch_train_loss, epoch_val_loss = (
        [],
        [],
        [],
        [],
        [],
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = running_correct = running_total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_acc, val_ura, _ = evaluate(dev_loader, sig_dev, unseen_dev_sigs, model)

        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_val_acc.append(val_acc)
        epoch_val_ura.append(val_ura)
        epoch_val_loss.append(0.0)  # placeholder, val loss not computed

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} URA={val_ura:.3f}"
        )

    # evaluation on test set
    test_acc, test_ura, test_preds = evaluate(
        test_loader, sig_test, unseen_test_sigs, model
    )
    print(f"Test accuracy={test_acc:.3f}  Test URA={test_ura:.3f}")

    # store metrics
    ed = experiment_data["batch_size_tuning"]["SPR_BENCH"]
    ed["metrics"]["train_acc"].append(epoch_train_acc)
    ed["metrics"]["val_acc"].append(epoch_val_acc)
    ed["metrics"]["val_ura"].append(epoch_val_ura)
    ed["metrics"]["test_acc"].append(test_acc)
    ed["metrics"]["test_ura"].append(test_ura)
    ed["losses"]["train"].append(epoch_train_loss)
    ed["losses"]["val"].append(epoch_val_loss)
    ed["predictions"].append(test_preds)
    ed["timestamps"].append(time.time())

# ------------------- save experiment data -------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
