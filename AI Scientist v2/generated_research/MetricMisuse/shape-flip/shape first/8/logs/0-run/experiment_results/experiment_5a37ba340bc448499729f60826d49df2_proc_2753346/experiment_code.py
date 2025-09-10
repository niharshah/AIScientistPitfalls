# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, string, time, json, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ----------------- reproducibility -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

# ----------------- feature encoding -----------------
shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3  # shape hist + colour hist + {seq_len,varieties}


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


# encode once
X_train, y_train, sig_train = encode_dataset(dsets["train"])
X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
X_test, y_test, sig_test = encode_dataset(dsets["test"])


class SPRTorchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


train_loader_full = DataLoader(
    SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True
)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=256)
test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=256)


# ----------------- model -----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- evaluation util -----------------
def eval_loader(model, loader, sigs_all, unseen_signatures):
    model.eval()
    correct = total = correct_unseen = total_unseen = 0
    preds_all = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            preds_all.extend(preds.cpu().numpy())
            total += y.size(0)
            correct += (preds == y).sum().item()
            for p, y_true in zip(preds.cpu().numpy(), y.cpu().numpy()):
                sig = sigs_all[idx]
                if sig in unseen_signatures:
                    total_unseen += 1
                    if p == y_true:
                        correct_unseen += 1
                idx += 1
    acc = correct / total
    ura = correct_unseen / total_unseen if total_unseen else 0.0
    return acc, ura, preds_all


# Identify unseen signatures
train_signatures = set(sig_train)
unseen_dev_sigs = {s for s in sig_dev if s not in train_signatures}
unseen_test_sigs = {s for s in sig_test if s not in train_signatures}

# ----------------- hyper-parameter tuning -----------------
epoch_options = [5, 10, 20, 30]
experiment_data = {"EPOCHS": {}}

for num_epochs in epoch_options:
    print(f"\n=== Training with {num_epochs} epochs ===")
    model = MLP(feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # prepare fresh data loaders each run (shuffling matters)
    train_loader = DataLoader(SPRTorchDS(X_train, y_train), batch_size=64, shuffle=True)

    run_key = f"epochs_{num_epochs}"
    experiment_data["EPOCHS"][run_key] = {
        "metrics": {"train_acc": [], "val_acc": [], "val_ura": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "timestamps": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        run_loss = run_corr = run_total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_corr += (logits.argmax(dim=1) == y).sum().item()
            run_total += y.size(0)
        train_loss = run_loss / run_total
        train_acc = run_corr / run_total
        val_acc, val_ura, _ = eval_loader(model, dev_loader, sig_dev, unseen_dev_sigs)

        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} URA={val_ura:.3f}"
        )

        ed = experiment_data["EPOCHS"][run_key]
        ed["losses"]["train"].append(train_loss)
        ed["metrics"]["train_acc"].append(train_acc)
        ed["metrics"]["val_acc"].append(val_acc)
        ed["metrics"]["val_ura"].append(val_ura)
        ed["timestamps"].append(time.time())

    # ----- final test evaluation -----
    test_acc, test_ura, test_preds = eval_loader(
        model, test_loader, sig_test, unseen_test_sigs
    )
    print(f"Test  acc={test_acc:.3f}  Test URA={test_ura:.3f}")

    experiment_data["EPOCHS"][run_key]["predictions"] = test_preds
    experiment_data["EPOCHS"][run_key]["test_acc"] = test_acc
    experiment_data["EPOCHS"][run_key]["test_ura"] = test_ura

# ----------------- save -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
