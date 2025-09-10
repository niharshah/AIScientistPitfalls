import os, pathlib, random, time, json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# Working directory & GPU/CPU handling
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# Utility to load SPR_BENCH (copied from spec)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.strip().split() if len(tok) > 1))


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


def dwhs(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) > 0 else 0.0


# ------------------------------------------------------------------
# Try to load real data; if unavailable, create small synthetic set
def make_synthetic(n_train=200, n_dev=50, n_test=50) -> DatasetDict:
    shapes = list("ABCDE")
    colors = [str(c) for c in range(5)]

    def rand_seq():
        length = random.randint(5, 15)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(length)]
        return " ".join(toks)

    def make_split(n):
        seqs = [rand_seq() for _ in range(n)]
        labels = [random.choice(["X", "Y", "Z"]) for _ in range(n)]
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    d = DatasetDict()
    d["train"] = load_dataset(
        "json", data_files={"train": [json.dumps(x) for x in []]}, split="train"
    )  # placeholder
    # workaround to build dataset quickly using HF "datasets"; use list to dataset
    d["train"] = load_dataset("json", data_files={"train": []}, split="train")
    d["dev"] = d["test"] = d["train"]  # Initialise
    # replace with arrow table
    from datasets import Dataset as HFDataset

    d["train"] = HFDataset.from_dict(make_split(n_train))
    d["dev"] = HFDataset.from_dict(make_split(n_dev))
    d["test"] = HFDataset.from_dict(make_split(n_test))
    return d


def get_dataset() -> DatasetDict:
    try:
        root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        if not root.exists():
            raise FileNotFoundError
        d = load_spr_bench(root)
        print("Loaded real SPR_BENCH dataset")
    except Exception as e:
        print("Falling back to synthetic dataset because:", e)
        d = make_synthetic()
    return d


spr_bench = get_dataset()


# ------------------------------------------------------------------
# Glyph vocabulary and clustering
def build_glyph_clusters(train_sequences: List[str], n_clusters: int = 20):
    glyphs = set()
    for seq in train_sequences:
        glyphs.update(seq.strip().split())
    glyphs = sorted(list(glyphs))
    shapes = sorted(list(set(g[0] for g in glyphs)))
    colors = sorted(list(set(g[1:] for g in glyphs)))
    shape2id = {s: i for i, s in enumerate(shapes)}
    color2id = {c: i for i, c in enumerate(colors)}
    features = np.array([[shape2id[g[0]], color2id[g[1:]]] for g in glyphs])
    k = min(n_clusters, len(glyphs))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    cluster_ids = km.fit_predict(features)
    glyph2cluster = {g: c for g, c in zip(glyphs, cluster_ids)}
    return glyph2cluster, k, shape2id, color2id


glyph2cluster, n_clusters, shape2id, color2id = build_glyph_clusters(
    list(spr_bench["train"]["sequence"])
)


def sequence_to_hist(seq: str, cluster_map: Dict[str, int], k: int):
    counts = np.zeros(k, dtype=np.float32)
    tokens = seq.strip().split()
    if len(tokens) == 0:
        return counts
    for tok in tokens:
        if tok in cluster_map:
            counts[cluster_map[tok]] += 1
    counts = counts / len(tokens)
    return counts


# ------------------------------------------------------------------
# Prepare datasets (features + labels)
def prepare_split(hf_split, cluster_map, k):
    X = np.stack(
        [sequence_to_hist(s, cluster_map, k) for s in hf_split["sequence"]]
    ).astype(np.float32)
    y = np.array(hf_split["label"])
    seqs = list(hf_split["sequence"])
    return X, y, seqs


X_train, y_train, seq_train = prepare_split(
    spr_bench["train"], glyph2cluster, n_clusters
)
X_dev, y_dev, seq_dev = prepare_split(spr_bench["dev"], glyph2cluster, n_clusters)
X_test, y_test, seq_test = prepare_split(spr_bench["test"], glyph2cluster, n_clusters)

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_dev_enc = label_encoder.transform(y_dev)
y_test_enc = label_encoder.transform(y_test)
n_classes = len(label_encoder.classes_)


# ------------------------------------------------------------------
# Torch Dataset & DataLoader
class HistDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


batch_size = 128
train_loader = DataLoader(
    HistDataset(X_train, y_train_enc), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(
    HistDataset(X_dev, y_dev_enc), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    HistDataset(X_test, y_test_enc), batch_size=batch_size, shuffle=False
)


# ------------------------------------------------------------------
# Model
class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(n_clusters, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# Experiment data store
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------------------------------------------------------
# Training loop
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["y"].size(0)
    train_loss = total_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    # Evaluate on dev
    model.eval()
    total_loss = 0
    y_true_dev = []
    y_pred_dev = []
    with torch.no_grad():
        for batch in dev_loader:
            batch_t = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch_t["x"])
            loss = criterion(logits, batch_t["y"])
            total_loss += loss.item() * batch_t["y"].size(0)
            preds = logits.argmax(-1).cpu().numpy()
            y_pred_dev.extend(preds)
            y_true_dev.extend(batch_t["y"].cpu().numpy())
    val_loss = total_loss / len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    # Metrics
    y_true_lbl = label_encoder.inverse_transform(y_true_dev)
    y_pred_lbl = label_encoder.inverse_transform(y_pred_dev)
    cwa = color_weighted_accuracy(seq_dev, y_true_lbl, y_pred_lbl)
    swa = shape_weighted_accuracy(seq_dev, y_true_lbl, y_pred_lbl)
    val_dwhs = dwhs(cwa, swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "dwhs": val_dwhs}
    )
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"loss": train_loss})
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, CWA={cwa:.4f}, SWA={swa:.4f}, DWHS={val_dwhs:.4f}"
    )

# ------------------------------------------------------------------
# Final test evaluation
model.eval()
y_pred_test = []
with torch.no_grad():
    for batch in test_loader:
        batch_t = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = model(batch_t["x"])
        preds = logits.argmax(-1).cpu().numpy()
        y_pred_test.extend(preds)
y_pred_lbl_test = label_encoder.inverse_transform(y_pred_test)
cwa_test = color_weighted_accuracy(seq_test, y_test, y_pred_lbl_test)
swa_test = shape_weighted_accuracy(seq_test, y_test, y_pred_lbl_test)
dwhs_test = dwhs(cwa_test, swa_test)
print(f"Test  : CWA={cwa_test:.4f}, SWA={swa_test:.4f}, DWHS={dwhs_test:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred_lbl_test
experiment_data["SPR_BENCH"]["ground_truth"] = y_test
experiment_data["SPR_BENCH"]["test_metrics"] = {
    "cwa": cwa_test,
    "swa": swa_test,
    "dwhs": dwhs_test,
}

# ------------------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir,"experiment_data.npy")}')
