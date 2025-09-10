# -------------------------------------------------
# Symbolic Glyph Clustering – bug-fixed experiment
# -------------------------------------------------
import os, pathlib, random, time, json, math

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# 1.  Device handling (MUST come before constructing optimizers etc.)
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# 2.  Helper: locate SPR_BENCH folder no matter where we are launched
# ------------------------------------------------------------------
def locate_spr_bench() -> pathlib.Path:
    """
    Search for SPR_BENCH folder by:
    (1) environment variable SPR_BENCH_DIR
    (2) walking up parent directories from cwd
    Returns absolute pathlib.Path if found, else raises FileNotFoundError.
    """
    env = os.getenv("SPR_BENCH_DIR")
    if env:
        p = pathlib.Path(env).expanduser()
        if (p / "train.csv").exists():
            return p.resolve()

    # walk up the directory tree
    cwd = pathlib.Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "SPR_BENCH"
        if (candidate / "train.csv").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find SPR_BENCH dataset. "
        "Set SPR_BENCH_DIR env variable or place SPR_BENCH folder "
        "containing train.csv/dev.csv/test.csv somewhere above the run directory."
    )


DATA_PATH = locate_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ------------------------------------------------------------------
# 3.  Dataset loading utilities
# ------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",  # treat the single csv as one split
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    ws = [count_color_variety(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(ws, y_true, y_pred)]
    return sum(corr) / sum(ws) if sum(ws) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    ws = [count_shape_variety(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(ws, y_true, y_pred)]
    return sum(corr) / sum(ws) if sum(ws) > 0 else 0.0


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) > 0 else 0.0


# ------------------------------------------------------------------
# 4.  Load dataset
# ------------------------------------------------------------------
spr = load_spr_bench(DATA_PATH)
print("Loaded splits sizes:", {k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# 5.  Analyse token space, build feature vectors
# ------------------------------------------------------------------
shapes, colors = set(), set()
for split in ["train", "dev"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.strip().split():
            if not tok:
                continue
            shapes.add(tok[0])
            colors.add(tok[1:])
shapes, colors = sorted(shapes), sorted(colors)
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}
feat_dim = len(shapes) + len(colors)

token_features, token_list = [], []
for s in shapes:
    for c in colors:
        tok = f"{s}{c}"
        vec = np.zeros(feat_dim, dtype=np.float32)
        vec[shape2idx[s]] = 1.0
        vec[len(shapes) + color2idx[c]] = 1.0
        token_features.append(vec)
        token_list.append(tok)
token_features = np.stack(token_features)

# ------------------------------------------------------------------
# 6.  K-means clustering on token features
# ------------------------------------------------------------------
K = 8
# scikit ≥1.4 allows n_init='auto', older versions need int
km = KMeans(n_clusters=K, random_state=0, n_init=10)
cluster_ids = km.fit_predict(token_features)
token2cluster = {tok: cid for tok, cid in zip(token_list, cluster_ids)}
print("Finished KMeans clustering of tokens.")


def seq_to_hist(seq: str) -> np.ndarray:
    hist = np.zeros(K, dtype=np.float32)
    for tok in seq.strip().split():
        cid = token2cluster.get(tok, -1)
        if cid >= 0:
            hist[cid] += 1.0
    # normalise histogram to unit length to keep scale consistent
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


# ------------------------------------------------------------------
# 7.  Label mappings
# ------------------------------------------------------------------
all_labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(all_labels)}
idx2label = {i: l for l, i in label2idx.items()}
num_labels = len(all_labels)


# ------------------------------------------------------------------
# 8.  PyTorch Dataset & DataLoaders
# ------------------------------------------------------------------
class SPRGlyphDataset(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        hist = seq_to_hist(self.seqs[idx])
        label = label2idx[self.labels[idx]]
        return {
            "x": torch.tensor(hist, dtype=torch.float32),
            "y": torch.tensor(label, dtype=torch.long),
            "seq": self.seqs[idx],
        }


batch_size = 128
train_dl = DataLoader(SPRGlyphDataset("train"), batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(SPRGlyphDataset("dev"), batch_size=batch_size, shuffle=False)
test_dl = DataLoader(SPRGlyphDataset("test"), batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------------
# 9.  Model, criterion, optimizer
# ------------------------------------------------------------------
model = nn.Linear(K, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# ------------------------------------------------------------------
# 10.  Data structure for logging / saving
# ------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------
# 11.  Training loop
# ------------------------------------------------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    # ---- Train ----
    model.train()
    running_loss = 0.0
    for batch in train_dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["x"].size(0)
    epoch_train_loss = running_loss / len(train_dl.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, epoch_train_loss))

    # ---- Validation ----
    model.eval()
    val_loss, all_pred, all_true, all_seq = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_dl:
            seqs = batch["seq"]
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            val_loss += loss.item() * batch["x"].size(0)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            trues = batch["y"].cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(trues)
            all_seq.extend(seqs)
    val_loss /= len(dev_dl.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))

    pred_labels = [idx2label[p] for p in all_pred]
    true_labels = [idx2label[t] for t in all_true]
    cwa = color_weighted_accuracy(all_seq, true_labels, pred_labels)
    swa = shape_weighted_accuracy(all_seq, true_labels, pred_labels)
    hcsa = harmonic_csa(cwa, swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, hcsa))

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"CWA={cwa:.3f} SWA={swa:.3f} HCSA={hcsa:.3f}"
    )

# ------------------------------------------------------------------
# 12.  Final Test Evaluation
# ------------------------------------------------------------------
model.eval()
test_preds, test_true, test_seq = [], [], []
with torch.no_grad():
    for batch in test_dl:
        seqs = batch["seq"]
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["x"])
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        trues = batch["y"].cpu().tolist()
        test_preds.extend(preds)
        test_true.extend(trues)
        test_seq.extend(seqs)

test_pred_labels = [idx2label[p] for p in test_preds]
test_true_labels = [idx2label[t] for t in test_true]
cwa_test = color_weighted_accuracy(test_seq, test_true_labels, test_pred_labels)
swa_test = shape_weighted_accuracy(test_seq, test_true_labels, test_pred_labels)
hcsa_test = harmonic_csa(cwa_test, swa_test)

print(f"Test  CWA={cwa_test:.3f} SWA={swa_test:.3f} HCSA={hcsa_test:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = test_pred_labels
experiment_data["SPR_BENCH"]["ground_truth"] = test_true_labels
experiment_data["SPR_BENCH"]["metrics"]["test"] = hcsa_test

# ------------------------------------------------------------------
# 13.  Save experiment data
# ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
