import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from collections import Counter, defaultdict

# ---------- work dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "cluster_silhouette": {},
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helpers ----------
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _load(n):  # each csv is treated as a split
        return load_dataset(
            "csv", data_files=str(root / n), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_color(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape(seq):
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y, p):
    w = [count_color(s) for s in seqs]
    return sum(wi for wi, t, pp in zip(w, y, p) if t == pp) / max(sum(w), 1)


def swa(seqs, y, p):
    w = [count_shape(s) for s in seqs]
    return sum(wi for wi, t, pp in zip(w, y, p) if t == pp) / max(sum(w), 1)


# ---------- dataset ----------
PATH = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(PATH).exists():
    dsets = load_spr(pathlib.Path(PATH))
else:
    # small synthetic fallback
    def synth(n):
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
        lbls = np.random.choice(list("ABC"), size=n).tolist()
        return {"sequence": seqs, "label": lbls}

    dsets = DatasetDict(
        {
            s: load_dataset("json", data_files={"train": synth(n)}, split="train")
            for s, n in zip(["train", "dev", "test"], [2000, 500, 500])
        }
    )

labels = sorted(set(dsets["train"]["label"]))
lid = {l: i for i, l in enumerate(labels)}
y_train = np.array([lid[l] for l in dsets["train"]["label"]], dtype=np.int64)
y_val = np.array([lid[l] for l in dsets["dev"]["label"]], dtype=np.int64)
y_test = np.array([lid[l] for l in dsets["test"]["label"]], dtype=np.int64)

# ---------- glyph clustering ----------
train_tokens = [tok for seq in dsets["train"]["sequence"] for tok in seq.split()]
uniq_tokens = sorted(set(train_tokens))
# simple 2-d embedding: shape id , color id
shape_ids = {s: i for i, s in enumerate(sorted({t[0] for t in uniq_tokens}))}
color_ids = {c: i for i, c in enumerate(sorted({t[1] for t in uniq_tokens}))}
emb = np.array([[shape_ids[t[0]], color_ids[t[1]]] for t in uniq_tokens], dtype=float)
n_clusters = min(8, len(uniq_tokens) // 2) if len(uniq_tokens) >= 4 else 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(emb)
tok2cluster = {tok: int(kmeans.labels_[i]) for i, tok in enumerate(uniq_tokens)}

sil_samples = (
    silhouette_samples(emb, kmeans.labels_)
    if n_clusters > 1
    else np.ones(len(uniq_tokens))
)
cluster_sil = {}
for idx, c in enumerate(kmeans.labels_):
    cluster_sil.setdefault(c, []).append(sil_samples[idx])
cluster_sil = {c: float(np.mean(v)) for c, v in cluster_sil.items()}
experiment_data["SPR_BENCH"]["cluster_silhouette"] = cluster_sil


def seq_to_cluster_tokens(seq: str) -> str:
    cls = [tok2cluster.get(t, 0) for t in seq.split()]
    return " ".join(map(str, cls))


def majority_cluster(seq: str) -> int:
    cls = [tok2cluster.get(t, 0) for t in seq.split()]
    return Counter(cls).most_common(1)[0][0] if cls else 0


train_cluster_str = [seq_to_cluster_tokens(s) for s in dsets["train"]["sequence"]]
val_cluster_str = [seq_to_cluster_tokens(s) for s in dsets["dev"]["sequence"]]
test_cluster_str = [seq_to_cluster_tokens(s) for s in dsets["test"]["sequence"]]
train_major = [majority_cluster(s) for s in dsets["train"]["sequence"]]
val_major = [majority_cluster(s) for s in dsets["dev"]["sequence"]]
test_major = [majority_cluster(s) for s in dsets["test"]["sequence"]]

# ---------- vectoriser ----------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vectorizer.fit(train_cluster_str)
X_train = vectorizer.transform(train_cluster_str).astype(np.float32).toarray()
X_val = vectorizer.transform(val_cluster_str).astype(np.float32).toarray()
X_test = vectorizer.transform(test_cluster_str).astype(np.float32).toarray()


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, inp, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 256), nn.ReLU(), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.net(x)


model = MLP(X_train.shape[1], len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=128,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=256
)


def ccwa(preds, y_true, major_clusters):
    correct = [int(p == t) for p, t in zip(preds, y_true)]
    cluster_correct = defaultdict(list)
    for c, cor in zip(major_clusters, correct):
        cluster_correct[c].append(cor)
    num, den = 0.0, 0.0
    for c, vals in cluster_correct.items():
        Ai = float(np.mean(vals))
        Si = cluster_sil.get(c, 0.0)
        num += Si * Ai
        den += Si
    return num / den if den > 0 else 0.0


# ---------- training ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # validation
    model.eval()
    v_loss = 0.0
    preds = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            v_loss += loss.item() * xb.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
    v_loss /= len(val_loader.dataset)
    acc = float(np.mean(np.array(preds) == y_val))
    cwa_val = cwa(dsets["dev"]["sequence"], y_val, preds)
    swa_val = swa(dsets["dev"]["sequence"], y_val, preds)
    ccwa_val = ccwa(preds, y_val, val_major)

    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa_val, "swa": swa_val, "ccwa": ccwa_val}
    )

    print(
        f"Epoch {epoch}: validation_loss = {v_loss:.4f} | ACC={acc:.3f} CWA={cwa_val:.3f} SWA={swa_val:.3f} CCWA={ccwa_val:.3f}"
    )

# ---------- test ----------
with torch.no_grad():
    logits = model(torch.from_numpy(X_test).to(device))
test_preds = logits.argmax(1).cpu().numpy()
test_acc = float(np.mean(test_preds == y_test))
test_cwa = cwa(dsets["test"]["sequence"], y_test, test_preds)
test_swa = swa(dsets["test"]["sequence"], y_test, test_preds)
test_ccwa = ccwa(test_preds, y_test, test_major)
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "ccwa": test_ccwa,
}
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = y_test
print(
    f"\nTEST  ACC={test_acc:.3f} CWA={test_cwa:.3f} SWA={test_swa:.3f} CCWA={test_ccwa:.3f}"
)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
