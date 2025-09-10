import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from collections import defaultdict
from typing import List, Dict

# ------------------------------------------------------------------
# mandatory working dir & experiment storage
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"val": [], "test": None},
        "predictions": {"val": None, "test": None},
        "ground_truth": {"val": None, "test": None},
        "silhouette": None,
    }
}

# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    data = DatasetDict()
    for split in ["train", "dev", "test"]:
        data[split] = _load(f"{split}.csv")
    return data


# Fallback synthetic data when benchmark not present
BENCH_PATH = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(BENCH_PATH).exists():
    dsets = load_spr_bench(pathlib.Path(BENCH_PATH))
else:
    print("SPR_BENCH not found – creating small synthetic toy data")

    def _synth(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        seqs = [
            " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(4, 9),
                )
            )
            for _ in range(n)
        ]
        labels = np.random.choice(list("ABC"), size=n).tolist()
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 60, 60]):
        tmp_file = os.path.join(working_dir, f"{split}.json")
        np.save(tmp_file, _synth(n))  # placeholder – not actually used by HF
        dsets[split] = load_dataset(
            "json", data_files={"train": _synth(n)}, split="train"
        )


# ------------------------------------------------------------------
# traditional metrics
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)) / max(
        sum(w), 1
    )


# ------------------------------------------------------------------
# 1) collect unique glyphs
all_seqs = (
    dsets["train"]["sequence"] + dsets["dev"]["sequence"] + dsets["test"]["sequence"]
)
glyphs = sorted({tok for seq in all_seqs for tok in seq.strip().split()})
shapes = sorted({g[0] for g in glyphs})
colors = sorted({g[1] for g in glyphs if len(g) > 1})
shape2id = {s: i for i, s in enumerate(shapes)}
color2id = {c: i for i, c in enumerate(colors)}


def glyph_vec(g):
    # 2-dim embedding: shape id, color id
    return np.array([shape2id[g[0]], color2id.get(g[1], -1)], dtype=np.float32)


X_glyph = np.vstack([glyph_vec(g) for g in glyphs])

# 2) cluster glyphs
n_clusters = min(8, len(glyphs))  # small
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_glyph)
glyph2cluster = {g: int(c) for g, c in zip(glyphs, kmeans.labels_)}

# silhouette per glyph then per cluster
sil_samples = (
    silhouette_samples(X_glyph, kmeans.labels_)
    if len(set(kmeans.labels_)) > 1
    else np.ones(len(glyphs))
)
cluster_sil = defaultdict(list)
for g, sil in zip(glyphs, sil_samples):
    cluster_sil[glyph2cluster[g]].append(float(sil))
cluster_sil = {cid: float(np.mean(v)) for cid, v in cluster_sil.items()}
experiment_data["SPR_BENCH"]["silhouette"] = cluster_sil


# ------------------------------------------------------------------
# helper to convert original sequences -> cluster sequences (string)
def seq_to_cluster_string(seq: str) -> str:
    return " ".join(str(glyph2cluster.get(tok, -1)) for tok in seq.split())


train_cluster_seqs = [seq_to_cluster_string(s) for s in dsets["train"]["sequence"]]
val_cluster_seqs = [seq_to_cluster_string(s) for s in dsets["dev"]["sequence"]]
test_cluster_seqs = [seq_to_cluster_string(s) for s in dsets["test"]["sequence"]]

# labels
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
y_tr = np.array([label2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], np.int64)
y_te = np.array([label2id[l] for l in dsets["test"]["label"]], np.int64)
num_classes = len(labels)

# ------------------------------------------------------------------
# vectorize cluster sequences
vectorizer = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vectorizer.fit(train_cluster_seqs)
Xtr = vectorizer.transform(train_cluster_seqs).astype(np.float32).toarray()
Xval = vectorizer.transform(val_cluster_seqs).astype(np.float32).toarray()
Xte = vectorizer.transform(test_cluster_seqs).astype(np.float32).toarray()


# ------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, n_cls)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleMLP(Xtr.shape[1], num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_tr)),
    batch_size=128,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xval), torch.from_numpy(y_val)), batch_size=256
)


# ------------------------------------------------------------------
def compute_ccwa(sequences: List[str], preds: np.ndarray, gold: np.ndarray) -> float:
    correct = preds == gold
    cluster_seq_map: Dict[int, List[int]] = defaultdict(list)
    for idx, seq in enumerate(sequences):
        present = set(int(tok) for tok in seq.split() if tok)
        for cid in present:
            cluster_seq_map[cid].append(idx)
    num, den = 0.0, 0.0
    for cid, idxs in cluster_seq_map.items():
        if cid not in cluster_sil:
            continue
        sil = cluster_sil[cid]
        if sil <= 0:
            continue
        acc = np.mean(correct[idxs])
        num += sil * acc
        den += sil
    return num / den if den > 0 else 0.0


# ------------------------------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    # ---- train
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # ---- validate
    model.eval()
    val_loss, preds = 0.0, []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
    val_loss /= len(val_loader.dataset)
    preds = np.array(preds)
    acc = np.mean(preds == y_val)
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
    ccwa = compute_ccwa(val_cluster_seqs, preds, y_val)

    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {
            "epoch": epoch,
            "acc": float(acc),
            "cwa": float(cwa),
            "swa": float(swa),
            "ccwa": float(ccwa),
        }
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}  "
        f"ACC={acc:.3f}  CWA={cwa:.3f}  SWA={swa:.3f}  CCWA={ccwa:.3f}"
    )

# ------------------------------------------------------------------
# final test evaluation
model.eval()
with torch.no_grad():
    test_logits = model(torch.from_numpy(Xte).to(device))
test_preds = test_logits.argmax(1).cpu().numpy()

test_acc = np.mean(test_preds == y_te)
test_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y_te, test_preds)
test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y_te, test_preds)
test_ccwa = compute_ccwa(test_cluster_seqs, test_preds, y_te)

experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "acc": float(test_acc),
    "cwa": float(test_cwa),
    "swa": float(test_swa),
    "ccwa": float(test_ccwa),
}
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = y_te.tolist()

print(
    "\nTEST  ACC={:.3f}  CWA={:.3f}  SWA={:.3f}  CCWA={:.3f}".format(
        test_acc, test_cwa, test_swa, test_ccwa
    )
)

# ------------------------------------------------------------------
# save everything
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All metrics saved to", os.path.join(working_dir, "experiment_data.npy"))
