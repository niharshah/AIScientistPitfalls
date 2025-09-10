import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples
from datasets import load_dataset, DatasetDict
from collections import defaultdict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"val": [], "test": {}},
        "predictions": [],
        "ground_truth": [],
        "clusters": {},
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- utility from baseline ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict()
    for sp in ["train", "dev", "test"]:
        ds[sp] = _load(f"{sp}.csv")
    return ds


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred))
    return num / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred))
    return num / max(sum(w), 1)


# ---------- data ----------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
if DATA_PATH.exists():
    dsets = load_spr_bench(DATA_PATH)
else:
    # create small synthetic fallback
    print("Warning: SPR_BENCH not found → generating synthetic toy data.")
    shapes, colors = list("▲●■"), list("rgb")

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            seq = " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(4, 10),
                )
            )
            seqs.append(seq)
            labels.append(np.random.choice(list("ABC")))
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for sp, n in [("train", 500), ("dev", 100), ("test", 200)]:
        tmp_path = os.path.join(working_dir, f"{sp}.json")
        np.save(tmp_path, synth(n))
        dsets[sp] = load_dataset("json", data_files=tmp_path, split="train")

# ---------- label encoding ----------
labels = sorted(set(dsets["train"]["label"]))
l2i = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
y_train = np.array([l2i[l] for l in dsets["train"]["label"]], dtype=np.int64)
y_val = np.array([l2i[l] for l in dsets["dev"]["label"]], dtype=np.int64)
y_test = np.array([l2i[l] for l in dsets["test"]["label"]], dtype=np.int64)

# ---------- 1. glyph clustering ----------
# gather all tokens in training set
tokens = [tok for seq in dsets["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in tokens})
colors = sorted({t[1] for t in tokens})
shape2id = {s: i for i, s in enumerate(shapes)}
color2id = {c: i for i, c in enumerate(colors)}

X_tok = np.array([[shape2id[t[0]], color2id[t[1]]] for t in tokens], dtype=float)
n_clusters = min(20, len(set(tokens)))  # small k to keep silhouette meaningful
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_tok)
tok2cluster = {}
idx = 0
for seq in dsets["train"]["sequence"]:
    for tok in seq.split():
        if tok not in tok2cluster:
            tok2cluster[tok] = kmeans.labels_[idx]
        idx += 1
# silhouette coefficients
sil_vals = silhouette_samples(X_tok, kmeans.labels_)
cluster_sil = defaultdict(list)
for lbl, val in zip(kmeans.labels_, sil_vals):
    cluster_sil[lbl].append(val)
cluster_si = {c: float(np.mean(v)) for c, v in cluster_sil.items()}
experiment_data["SPR_BENCH"]["clusters"]["silhouette"] = cluster_si


def seq_to_cluster_string(seq: str) -> str:
    return " ".join(str(tok2cluster.get(tok, -1)) for tok in seq.split())


train_cluster_seq = [seq_to_cluster_string(s) for s in dsets["train"]["sequence"]]
val_cluster_seq = [seq_to_cluster_string(s) for s in dsets["dev"]["sequence"]]
test_cluster_seq = [seq_to_cluster_string(s) for s in dsets["test"]["sequence"]]

# ---------- 2. feature extraction ----------
vect = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vect.fit(train_cluster_seq)


def vec(list_seq):
    return vect.transform(list_seq).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vec, [train_cluster_seq, val_cluster_seq, test_cluster_seq]
)


# ---------- 3. model ----------
class MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 256), nn.ReLU(), nn.Linear(256, out))

    def forward(self, x):
        return self.net(x)


model = MLP(X_train.shape[1], num_classes).to(device)
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


# ---------- CCWA helper ----------
def compute_ccwa(seqs, y_true, y_pred):
    # Per-cluster accuracy
    cluster_total = defaultdict(int)
    cluster_corr = defaultdict(int)
    for seq, yt, yp in zip(seqs, y_true, y_pred):
        clust_ids = set(int(cid) for cid in seq.split())
        for cid in clust_ids:
            cluster_total[cid] += 1
            if yt == yp:
                cluster_corr[cid] += 1
    num, den = 0.0, 0.0
    for cid, total in cluster_total.items():
        Si = cluster_si.get(cid, 0.0)
        Ai = cluster_corr[cid] / total
        num += Si * Ai
        den += Si
    return num / den if den > 0 else 0.0


# ---------- 4. training loop ----------
epochs = 6
for epoch in range(1, epochs + 1):
    # train
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    tr_loss = running_loss / len(train_loader.dataset)

    # validate
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
    acc = (np.array(preds) == y_val).mean()
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
    ccwa = compute_ccwa(val_cluster_seq, y_val, preds)

    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": ccwa}
    )
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CCWA={ccwa:.3f}"
    )

# ---------- 5. test evaluation ----------
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(X_test).to(device))
    test_preds = logits.argmax(1).cpu().numpy()

test_acc = (test_preds == y_test).mean()
test_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
test_ccwa = compute_ccwa(test_cluster_seq, y_test, test_preds)

experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "ccwa": test_ccwa,
}
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = y_test

print(
    f"\nTEST → ACC={test_acc:.3f} CWA={test_cwa:.3f} SWA={test_swa:.3f} CCWA={test_ccwa:.3f}"
)

# ---------- 6. save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data.")
