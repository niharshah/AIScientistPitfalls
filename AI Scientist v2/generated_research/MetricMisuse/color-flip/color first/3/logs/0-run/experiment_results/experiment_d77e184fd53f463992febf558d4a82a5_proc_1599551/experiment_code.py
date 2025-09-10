import os, pathlib, random, time, json, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ---------- working directory ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- GPU / CPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- helper: robust data path finder ----------
def find_spr_data_path() -> pathlib.Path:
    """Return a path containing train.csv/dev.csv/test.csv for SPR_BENCH"""
    candidate_strs = [
        os.environ.get("SPR_DATA_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in candidate_strs:
        if not p:
            continue
        root = pathlib.Path(p).expanduser().resolve()
        if (
            (root / "train.csv").is_file()
            and (root / "dev.csv").is_file()
            and (root / "test.csv").is_file()
        ):
            print(f"Dataset found at: {root}")
            return root
    raise FileNotFoundError(
        "Could not locate SPR_BENCH dataset. "
        "Please place train.csv/dev.csv/test.csv in one of the default locations "
        "or set environment variable SPR_DATA_PATH."
    )


DATA_PATH = find_spr_data_path()


# ---------- dataset loader ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",  # treat csv as a single split
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- metric helpers ----------
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


# ---------- analyse tokens ----------
shapes, colors = set(), set()
for split in ["train", "dev"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.strip().split():
            if not tok:
                continue
            shapes.add(tok[0])
            colors.add(tok[1:])  # may be multi-char
shapes, colors = sorted(shapes), sorted(colors)
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}
feat_dim = len(shapes) + len(colors)

# ---------- build token feature matrix ----------
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

# ---------- K-means clustering ----------
K = 8
km = KMeans(n_clusters=K, random_state=0, n_init=10)
cluster_ids = km.fit_predict(token_features)
token2cluster = {tok: cid for tok, cid in zip(token_list, cluster_ids)}
print("Finished token clustering.")


# ---------- sequence -> normalised histogram ----------
def seq_to_hist(seq: str) -> np.ndarray:
    hist = np.zeros(K, dtype=np.float32)
    tokens = seq.strip().split()
    for tok in tokens:
        cid = token2cluster.get(tok, -1)
        if cid >= 0:
            hist[cid] += 1.0
    if len(tokens) > 0:
        hist /= len(tokens)  # normalise → frequencies
    return hist


# ---------- label maps ----------
all_labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(all_labels)}
idx2label = {i: l for l, i in label2idx.items()}
num_labels = len(all_labels)


# ---------- PyTorch dataset ----------
class SPRGlyphDataset(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(seq_to_hist(self.seqs[idx]), dtype=torch.float32),
            "y": torch.tensor(label2idx[self.labels[idx]], dtype=torch.long),
            "seq": self.seqs[idx],
        }


batch_size = 128
train_dl = DataLoader(SPRGlyphDataset("train"), batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(SPRGlyphDataset("dev"), batch_size=batch_size, shuffle=False)
test_dl = DataLoader(SPRGlyphDataset("test"), batch_size=batch_size, shuffle=False)

# ---------- simple linear classifier ----------
model = nn.Linear(K, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# ---------- training loop ----------
epochs = 10
for epoch in range(1, epochs + 1):
    # train
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
    train_loss = running_loss / len(train_dl.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))

    # validate
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

    # metrics
    pred_lbls = [idx2label[p] for p in all_pred]
    true_lbls = [idx2label[t] for t in all_true]
    cwa = color_weighted_accuracy(all_seq, true_lbls, pred_lbls)
    swa = shape_weighted_accuracy(all_seq, true_lbls, pred_lbls)
    hcsa = harmonic_csa(cwa, swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, hcsa))
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HCSA={hcsa:.3f}"
    )

# ---------- final test evaluation ----------
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

# ---------- save experiment ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
