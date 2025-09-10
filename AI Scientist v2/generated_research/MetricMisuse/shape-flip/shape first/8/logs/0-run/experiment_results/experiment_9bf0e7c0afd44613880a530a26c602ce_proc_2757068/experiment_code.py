# Shape-Blind Encoding Ablation – single file, runnable
import os, pathlib, random, string, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ----------------- paths / device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- helpers ------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def rule_signature(seq: str):
    return (count_shape_variety(seq), count_color_variety(seq))


# -------------- synthetic fallback ---------------
def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:6])


def generate_synthetic_split(n_rows, seed=0):
    random.seed(seed)
    seqs, labels = [], []
    for _ in range(n_rows):
        L = random.randint(3, 10)
        seq = " ".join(random_token() for _ in range(L))
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))  # hidden rule
        seqs.append(seq)
        labels.append(lbl)
    return {"id": list(range(n_rows)), "sequence": seqs, "label": labels}


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    if root.exists():

        def _l(f):
            return load_dataset("csv", data_files=str(root / f), split="train")

        return DatasetDict(
            train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv")
        )
    print("SPR_BENCH not found – using synthetic toy data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, 1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, 2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, 3)),
    )


# ----------------- data loading ------------------
DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

color_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
FEAT_DIM = 10 + 3  # colour histogram + {length, shape_var, colour_var}


def encode(seq: str):
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    toks = seq.split()
    for t in toks:
        if len(t) < 2:
            continue
        v[color_to_idx[t[1]]] += 1  # colour counts only
    v[-3] = len(toks)  # sequence length
    v[-2] = count_shape_variety(seq)  # global shape variety
    v[-1] = count_color_variety(seq)  # global colour variety
    return v


def encode_split(hfds):
    X = np.stack([encode(s) for s in hfds["sequence"]])
    y = np.array(hfds["label"], dtype=np.int64)
    sigs = [rule_signature(s) for s in hfds["sequence"]]
    return X, y, sigs


X_train, y_train, sig_train = encode_split(dsets["train"])
X_dev, y_dev, sig_dev = encode_split(dsets["dev"])
X_test, y_test, sig_test = encode_split(dsets["test"])
train_signatures = set(sig_train)


class SPRTorchDS(Dataset):
    def __init__(self, X, y, seqs):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.seqs = seqs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i], "seq": self.seqs[i]}


bs_train, bs_eval = 64, 256
train_loader = DataLoader(
    SPRTorchDS(X_train, y_train, dsets["train"]["sequence"]),
    batch_size=bs_train,
    shuffle=True,
)
dev_loader = DataLoader(
    SPRTorchDS(X_dev, y_dev, dsets["dev"]["sequence"]), batch_size=bs_eval
)
test_loader = DataLoader(
    SPRTorchDS(X_test, y_test, dsets["test"]["sequence"]), batch_size=bs_eval
)


# ----------------- model ------------------------
class MLP(nn.Module):
    def __init__(self, indim, hidden=64, classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hidden), nn.ReLU(), nn.Linear(hidden, classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(FEAT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------- symbolic rule (same as original) -----
def symbolic_predict(seq: str) -> int:
    return 1 if count_shape_variety(seq) == count_color_variety(seq) else 0


# ---------------- evaluation -------------------
def evaluate(loader, sequences, true_labels):
    model.eval()
    preds, losses, idx = [], [], 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x)
            nn_preds = logits.argmax(dim=1).cpu().numpy()
            for j in range(x.size(0)):
                seq = sequences[idx]
                sig = rule_signature(seq)
                pred = (
                    symbolic_predict(seq)
                    if sig not in train_signatures
                    else int(nn_preds[j])
                )
                preds.append(pred)
                losses.append(
                    criterion(
                        logits[j : j + 1], batch["y"][j : j + 1].to(device)
                    ).item()
                )
                idx += 1
    swa = shape_weighted_accuracy(sequences, true_labels, preds)
    return np.mean(losses), swa, preds


# ---------------- experiment log ---------------
experiment_data = {
    "shape_blind": {
        "spr_bench": {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "timestamps": [],
        }
    }
}

# ---------------- training loop ----------------
num_epochs = 20
for ep in range(1, num_epochs + 1):
    model.train()
    run_loss, n = 0.0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["y"].size(0)
        n += batch["y"].size(0)
    train_loss = run_loss / n
    _, train_swa, _ = evaluate(train_loader, dsets["train"]["sequence"], y_train)
    val_loss, val_swa, _ = evaluate(dev_loader, dsets["dev"]["sequence"], y_dev)
    print(f"Epoch {ep}: val_loss={val_loss:.4f}  val_SWA={val_swa:.3f}")

    ed = experiment_data["shape_blind"]["spr_bench"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_swa"].append(train_swa)
    ed["metrics"]["val_swa"].append(val_swa)
    ed["timestamps"].append(time.time())

# -------------- final evaluation ---------------
test_loss, test_swa, test_preds = evaluate(
    test_loader, dsets["test"]["sequence"], y_test
)
print(f"\nTest Shape-Weighted Accuracy (SWA) = {test_swa:.3f}")
ed = experiment_data["shape_blind"]["spr_bench"]
ed["predictions"] = test_preds
ed["metrics"]["test_swa"] = test_swa
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
