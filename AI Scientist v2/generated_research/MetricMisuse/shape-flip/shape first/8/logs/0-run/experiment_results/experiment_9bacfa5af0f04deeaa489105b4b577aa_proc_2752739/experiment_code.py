import os, pathlib, random, string, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- symbolic helpers ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def rule_signature(sequence: str):
    return (count_shape_variety(sequence), count_color_variety(sequence))


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
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))  # simple rule
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

# ---------- feature encoding ----------
shape_to_idx = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feature_dim = 26 + 10 + 3  # shape-hist + colour-hist + misc


def encode_sequence(seq: str) -> np.ndarray:
    vec = np.zeros(feature_dim, dtype=np.float32)
    for tok in seq.split():
        if len(tok) < 2:
            continue
        s, c = tok[0], tok[1]
        vec[shape_to_idx[s]] += 1
        vec[26 + colour_to_idx[c]] += 1
    vec[-3] = len(seq.split())
    vec[-2] = count_shape_variety(seq)
    vec[-1] = count_color_variety(seq)
    return vec


def encode_dataset(hf_ds):
    feats = np.stack([encode_sequence(s) for s in hf_ds["sequence"]])
    labels = np.array(hf_ds["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in hf_ds["sequence"]]
    return feats, labels, sigs


# ---------- torch dataset ----------
class SPRTorchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


# ---------- evaluation helper ----------
def evaluate(model, loader, sigs_all, unseen_sigs, criterion):
    model.eval()
    total = correct = total_unseen = correct_unseen = 0
    loss_sum = 0.0
    preds_all = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            total += y.size(0)
            correct += (preds == y).sum().item()
            # URA
            batch_sigs = sigs_all[idx : idx + y.size(0)]
            for p, t, sig in zip(preds.cpu().numpy(), y.cpu().numpy(), batch_sigs):
                if sig in unseen_sigs:
                    total_unseen += 1
                    if p == t:
                        correct_unseen += 1
            idx += y.size(0)
    acc = correct / total
    ura = correct_unseen / total_unseen if total_unseen else 0.0
    return loss_sum / total, acc, ura, preds_all


# ---------- hyper-parameter tuning set-up ----------
hidden_dims = [32, 64, 128, 256]
EPOCHS = 5
BATCH_TRAIN = 64
BATCH_EVAL = 256

experiment_data = {"hidden_dim": {"SPR_BENCH": {}}}  # tuning type  # dataset

for hd in hidden_dims:
    print(f"\n===== Training MLP with hidden_dim = {hd} =====")
    # prepare data loaders anew (datasets remain same)
    X_train, y_train, sig_train = encode_dataset(dsets["train"])
    X_dev, y_dev, sig_dev = encode_dataset(dsets["dev"])
    X_test, y_test, sig_test = encode_dataset(dsets["test"])

    train_loader = DataLoader(
        SPRTorchDS(X_train, y_train), batch_size=BATCH_TRAIN, shuffle=True
    )
    dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev), batch_size=BATCH_EVAL)
    test_loader = DataLoader(SPRTorchDS(X_test, y_test), batch_size=BATCH_EVAL)

    unseen_dev_sigs = {s for s in sig_dev if s not in set(sig_train)}
    unseen_test_sigs = {s for s in sig_test if s not in set(sig_train)}

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(feature_dim, hd, len(set(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage dict for this hidden dim
    h_store = {
        "metrics": {"train_acc": [], "val_acc": [], "val_ura": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = run_correct = run_total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_total += y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()

        train_loss = run_loss / run_total
        train_acc = run_correct / run_total

        val_loss, val_acc, val_ura, _ = evaluate(
            model, dev_loader, sig_dev, unseen_dev_sigs, criterion
        )

        print(
            f" Epoch {epoch}: train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  URA={val_ura:.3f}"
        )

        h_store["losses"]["train"].append(train_loss)
        h_store["losses"]["val"].append(val_loss)
        h_store["metrics"]["train_acc"].append(train_acc)
        h_store["metrics"]["val_acc"].append(val_acc)
        h_store["metrics"]["val_ura"].append(val_ura)
        h_store["timestamps"].append(time.time())

    # final test
    _, test_acc, test_ura, test_preds = evaluate(
        model, test_loader, sig_test, unseen_test_sigs, criterion
    )
    print(f" Final Test: acc={test_acc:.3f}  URA={test_ura:.3f}")
    h_store["metrics"]["test_acc"] = test_acc
    h_store["metrics"]["test_ura"] = test_ura
    h_store["predictions"] = test_preds

    experiment_data["hidden_dim"]["SPR_BENCH"][hd] = h_store

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
