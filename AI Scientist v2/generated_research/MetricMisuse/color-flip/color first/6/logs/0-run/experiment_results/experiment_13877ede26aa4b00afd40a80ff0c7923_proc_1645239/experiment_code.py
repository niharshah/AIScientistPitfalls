import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import math
import pathlib
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# ---------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------- Data loading ------------
try:
    from datasets import load_dataset, DatasetDict

    def load_spr_bench(root: pathlib.Path):
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

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
except Exception as e:
    # Fallback tiny synthetic dataset
    print("Could not load real dataset, generating synthetic toy data.")

    def synth_seq():
        shapes = list("ABCDE")
        colors = list("123")
        seq = " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 10))
        )
        return seq

    def synth_label(seq):
        # simple rule: even number of tokens -> class 0 else 1
        return 0 if len(seq.split()) % 2 == 0 else 1

    def make_split(n):
        return {
            "id": [f"x{i}" for i in range(n)],
            "sequence": [synth_seq() for _ in range(n)],
        }

    synth_train = make_split(500)
    synth_train["label"] = [synth_label(s) for s in synth_train["sequence"]]
    synth_dev = make_split(100)
    synth_dev["label"] = [synth_label(s) for s in synth_dev["sequence"]]
    synth_test = make_split(100)
    synth_test["label"] = [synth_label(s) for s in synth_test["sequence"]]
    spr = {"train": synth_train, "dev": synth_dev, "test": synth_test}


# ---------------- Metrics ----------------
def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def entropy_weighted_accuracy(seqs, y_true, y_pred):
    def entropy(seq):
        tokens = seq.strip().split()
        cnt = Counter(tokens)
        n = len(tokens)
        return -sum((c / n) * math.log2(c / n) for c in cnt.values())

    weights = [entropy(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# --------------- Glyph clustering --------
def tokenize(seq):
    return seq.strip().split()


train_seqs = (
    spr["train"]["sequence"]
    if isinstance(spr["train"], dict)
    else spr["train"]["sequence"]
)
all_tokens = set()
for s in train_seqs:
    all_tokens.update(tokenize(s))
all_tokens = sorted(all_tokens)


def glyph_features(tok):
    shape_ord = ord(tok[0].upper()) - ord("A")
    color_int = int(tok[1]) if len(tok) > 1 and tok[1].isdigit() else 0
    return np.array([shape_ord, color_int], dtype=float)


tok_feats = np.stack([glyph_features(t) for t in all_tokens])
n_clusters = min(10, len(all_tokens))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_ids = kmeans.fit_predict(tok_feats)
glyph2cluster = {tok: cid for tok, cid in zip(all_tokens, cluster_ids)}


def seq_to_hist(seq):
    vec = np.zeros(n_clusters, dtype=np.float32)
    for tok in tokenize(seq):
        cid = glyph2cluster.get(tok, 0)
        vec[cid] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec


# --------------- Torch Dataset -----------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"] if isinstance(split, dict) else split["sequence"]
        self.labels = split["label"] if isinstance(split, dict) else split["label"]
        self.hists = [seq_to_hist(s) for s in self.seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.hists[idx], dtype=torch.float32),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq": self.seqs[idx],
        }


train_ds = SPRDataset(spr["train"])
dev_ds = SPRDataset(spr["dev"])


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    seqs = [b["seq"] for b in batch]
    return {"x": xs, "y": ys, "seq": seqs}


train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate)

# --------------- Model -------------------
num_classes = len(set(train_ds.labels))
model = nn.Sequential(
    nn.Linear(n_clusters, 32), nn.ReLU(), nn.Linear(32, num_classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------ Experiment logging ---------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# --------------- Training loop -----------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["x"])
        loss = criterion(out, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["x"].size(0)
    train_loss = total_loss / len(train_ds)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    seq_collect = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            val_loss += loss.item() * batch["x"].size(0)
            preds = out.argmax(1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(batch["y"].cpu().numpy().tolist())
            seq_collect.extend(batch["seq"])
    val_loss /= len(dev_ds)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    cwa = color_weighted_accuracy(seq_collect, y_true, y_pred)
    swa = shape_weighted_accuracy(seq_collect, y_true, y_pred)
    ewa = entropy_weighted_accuracy(seq_collect, y_true, y_pred)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "ewa": ewa}
    )
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  CWA={cwa:.3f}  SWA={swa:.3f}  EWA={ewa:.3f}"
    )

# --------------- Plot & save -------------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. Metrics and losses saved to 'working' directory.")
