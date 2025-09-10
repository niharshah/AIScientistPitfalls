import os, pathlib, random, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# directory & device setup
# -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------
# Data-loading utilities copied from provided SPR.py snippet
# -----------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# -----------------------------------------------------------
# Dataset path (adapt if necessary)
# -----------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():  # fallback to a relative directory if running elsewhere
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr_bench = load_spr_bench(DATA_PATH)
print("Loaded SPR benchmark. Sizes:", {k: len(v) for k, v in spr_bench.items()})


# -----------------------------------------------------------
# Build glyph-shape clusters (latent groups)
# -----------------------------------------------------------
def get_shape(token: str):
    return token[0] if token else "?"


shape_set = set()
for seq in spr_bench["train"]["sequence"]:
    shape_set.update(get_shape(tok) for tok in seq.strip().split())
shape2idx = {s: i for i, s in enumerate(sorted(shape_set))}
num_clusters = len(shape2idx)
print(f"Identified {num_clusters} shape-based clusters.")


# -----------------------------------------------------------
# Vectorise sequences into count-of-cluster representations
# -----------------------------------------------------------
def seq_to_vec(seq: str):
    vec = np.zeros(num_clusters, dtype=np.float32)
    for tok in seq.strip().split():
        vec[shape2idx[get_shape(tok)]] += 1.0
    return vec


def vectorise_split(split_ds):
    X = np.stack([seq_to_vec(s) for s in split_ds["sequence"]])
    y = np.array(split_ds["label"])
    return X, y


X_train, y_train = vectorise_split(spr_bench["train"])
X_dev, y_dev = vectorise_split(spr_bench["dev"])
X_test, y_test = vectorise_split(spr_bench["test"])

# label encoding (make labels consecutive ints 0..C-1)
lbls = sorted(set(y_train))
lbl2idx = {l: i for i, l in enumerate(lbls)}
y_train = np.array([lbl2idx[l] for l in y_train])
y_dev = np.array([lbl2idx[l] for l in y_dev])
y_test = np.array([lbl2idx[l] for l in y_test])
num_classes = len(lbls)
print(f"Num classes = {num_classes}")


# -----------------------------------------------------------
# PyTorch dataset / loader
# -----------------------------------------------------------
class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


batch_size = 128
train_loader = DataLoader(
    VecDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(VecDataset(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(VecDataset(X_test, y_test), batch_size=batch_size)

# -----------------------------------------------------------
# Model definition
# -----------------------------------------------------------
model = nn.Sequential(
    nn.Linear(num_clusters, 64), nn.ReLU(), nn.Linear(64, num_classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------------------------------------
# experiment data logging structure
# -----------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        output = model(batch["x"])
        loss = criterion(output, batch["y"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["y"].size(0)
    train_loss = running_loss / len(train_loader.dataset)

    # evaluate on dev
    model.eval()
    dev_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            batch_cpu = batch  # keep seq alignment on cpu
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            dev_loss += loss.item() * batch["y"].size(0)
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred)
            gts.extend(batch_cpu["y"].numpy())
            seqs.extend(
                spr_bench["dev"]["sequence"][i * batch_size : (i + 1) * batch_size]
            )
    dev_loss /= len(dev_loader.dataset)

    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cpx = complexity_weighted_accuracy(seqs, gts, preds)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={dev_loss:.4f} | "
        f"CWA={cwa:.4f} | SWA={swa:.4f} | CpxWA={cpx:.4f}"
    )

    # log metrics
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(dev_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"CWA": cwa, "SWA": swa, "CpxWA": cpx}
    )

# -----------------------------------------------------------
# Final evaluation on test split
# -----------------------------------------------------------
model.eval()
preds, gts, seqs = [], [], []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        batch_cpu = batch
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(batch["x"])
        pred = out.argmax(1).cpu().numpy()
        preds.extend(pred)
        gts.extend(batch_cpu["y"].numpy())
        seqs.extend(
            spr_bench["test"]["sequence"][i * batch_size : (i + 1) * batch_size]
        )

cwa_test = color_weighted_accuracy(seqs, gts, preds)
swa_test = shape_weighted_accuracy(seqs, gts, preds)
cpx_test = complexity_weighted_accuracy(seqs, gts, preds)
print(
    f"TEST RESULTS  |  CWA={cwa_test:.4f} | SWA={swa_test:.4f} | CpxWA={cpx_test:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "CWA": cwa_test,
    "SWA": swa_test,
    "CpxWA": cpx_test,
}

# -----------------------------------------------------------
# Save experiment data
# -----------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all experiment data to", os.path.join(working_dir, "experiment_data.npy"))
