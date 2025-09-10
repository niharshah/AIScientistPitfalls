# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import List, Dict

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _ld(f"{split}.csv")
    return d


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("Dataset not found, building synthetic tiny split for demo")

    def synth(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        seqs = [
            " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            for _ in range(n)
        ]
        labels = np.random.choice(["A", "B", "C"], size=n).tolist()
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [400, 100, 100]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth(n)}, split="train"
        )


# ---------- helpers ----------
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ---------- BERT embedding for glyph tokens ----------
# try Transformers, fallback to simple ord vectors if offline
def embed_tokens(tokens: List[str]) -> np.ndarray:
    try:
        from transformers import AutoTokenizer, AutoModel

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        mdl = AutoModel.from_pretrained("bert-base-uncased").to(device)
        mdl.eval()
        with torch.no_grad():
            embs = []
            for t in tokens:
                inp = tok(t, return_tensors="pt").to(device)
                out = mdl(**inp).last_hidden_state[:, 0]  # CLS
                embs.append(out.squeeze(0).cpu().numpy())
        return np.vstack(embs)
    except Exception as e:
        print("Transformer load failed, using simple ord-embedding:", e)
        vec = []
        for t in tokens:
            a = ord(t[0])
            b = ord(t[1]) if len(t) > 1 else 0
            vec.append([a / 1000, b / 1000])  # small scale
        return np.array(vec, dtype=np.float32)


# ---------- build token clusters ----------
train_tokens = [tok for seq in dsets["train"]["sequence"] for tok in seq.split()]
uniq_tokens = sorted(set(train_tokens))
token_embs = embed_tokens(uniq_tokens)

n_clusters = min(8, max(2, len(uniq_tokens) // 3))
print(f"Clustering {len(uniq_tokens)} tokens into {n_clusters} clusters")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(token_embs)
tok2cluster = {t: int(c) for t, c in zip(uniq_tokens, kmeans.labels_)}
silh_vals = (
    silhouette_samples(token_embs, kmeans.labels_)
    if n_clusters > 1
    else np.ones(len(uniq_tokens))
)
cluster_silh = {i: np.mean(silh_vals[kmeans.labels_ == i]) for i in range(n_clusters)}
print(
    "Mean silhouette:",
    silhouette_score(token_embs, kmeans.labels_) if n_clusters > 1 else 1.0,
)


# ---------- dataset transformation ----------
def seq_to_cluster_string(seq: str) -> str:
    return " ".join(f"c{tok2cluster.get(tok,-1)}" for tok in seq.split())


cluster_strings = {
    split: [seq_to_cluster_string(s) for s in dsets[split]["sequence"]]
    for split in ["train", "dev", "test"]
}

# labels ids
labels = sorted(list(set(dsets["train"]["label"])))
lid = {l: i for i, l in enumerate(labels)}
y = {
    split: np.array([lid[l] for l in dsets[split]["label"]], dtype=np.int64)
    for split in ["train", "dev", "test"]
}

# ---------- vectorizers ----------
vec_token = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vec_clust = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))

vec_token.fit(dsets["train"]["sequence"])
vec_clust.fit(cluster_strings["train"])


def build_features(split: str) -> np.ndarray:
    X1 = vec_token.transform(dsets[split]["sequence"]).toarray().astype(np.float32)
    X2 = vec_clust.transform(cluster_strings[split]).toarray().astype(np.float32)
    return np.hstack([X1, X2])


X = {sp: build_features(sp) for sp in ["train", "dev", "test"]}
print("Feature dim:", X["train"].shape[1])


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(X["train"].shape[1], len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ---------- dataloaders ----------
def make_loader(split: str, bs=64):
    ds = TensorDataset(torch.from_numpy(X[split]), torch.from_numpy(y[split]))
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"))


loaders = {sp: make_loader(sp) for sp in ["train", "dev"]}


# ---------- CCWA ----------
def majority_cluster(seq_clusters: str) -> int:
    ids = [int(t[1:]) for t in seq_clusters.split() if t != "c-1"]
    return max(set(ids), key=ids.count) if ids else -1


def compute_ccwa(split: str, preds: np.ndarray) -> float:
    maj_clusters = np.array([majority_cluster(cs) for cs in cluster_strings[split]])
    ccwa_num, ccwa_den = 0.0, 0.0
    for cid in range(n_clusters):
        mask = maj_clusters == cid
        if not mask.any():
            continue
        Ai = (preds[mask] == y[split][mask]).mean()
        Si = cluster_silh.get(cid, 0)
        ccwa_num += Si * Ai
        ccwa_den += Si
    return ccwa_num / ccwa_den if ccwa_den > 0 else 0.0


# ---------- training loop ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in loaders["train"]:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(loaders["train"].dataset)

    model.eval()
    val_loss, val_preds = [], []
    with torch.no_grad():
        for xb, yb in loaders["dev"]:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss.append(loss.item() * xb.size(0))
            val_preds.extend(out.argmax(1).cpu().numpy())
    val_loss = sum(val_loss) / len(loaders["dev"].dataset)
    val_preds = np.array(val_preds)
    acc = (val_preds == y["dev"]).mean()
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], val_preds)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], val_preds)
    ccwa = compute_ccwa("dev", val_preds)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": ccwa}
    )

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} validation_loss = {val_loss:.4f} "
        f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CCWA={ccwa:.3f}"
    )

# ---------- test evaluation ----------
model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(X["test"]).to(device)).argmax(1).cpu().numpy()
test_acc = (preds == y["test"]).mean()
test_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y["test"], preds)
test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y["test"], preds)
test_ccwa = compute_ccwa("test", preds)
print(
    f"\nTEST  ACC={test_acc:.3f} CWA={test_cwa:.3f} SWA={test_swa:.3f} CCWA={test_ccwa:.3f}"
)

experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "ccwa": test_ccwa,
}
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = y["test"]

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
