import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Tuple, Any

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment container ----------
experiment_data: Dict[str, Any] = {
    "glyph_cluster_mlp": {
        "SPR_BENCH": {
            "losses": {"train": [], "val": []},
            "metrics": {"val": [], "test": None},
            "cluster_info": {},
            "predictions": [],
            "ground_truth": [],
            "sequences": [],
        }
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for sp in ["train", "dev", "test"]:
        out[sp] = _load(f"{sp}.csv")
    return out


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    # small synthetic fallback (rarely triggered in evaluation env)
    print("SPR_BENCH missing, generating synthetic toy data.")
    shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]

    def make_split(n):
        seqs = [
            " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(4, 10),
                )
            )
            for _ in range(n)
        ]
        labels = np.random.choice(["A", "B", "C"], size=n).tolist()
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for sp, n in [("train", 200), ("dev", 50), ("test", 50)]:
        dsets[sp] = load_dataset(
            "json", data_files={"train": make_split(n)}, split="train"
        )


# ---------- helpers ----------
def split_token(tok: str) -> Tuple[str, str]:
    return (tok[0], tok[1]) if len(tok) >= 2 else (tok, "")  # safety


def collect_shapes_colors(seqs: List[str]):
    shapes, colors = set(), set()
    for seq in seqs:
        for tok in seq.split():
            sh, co = split_token(tok)
            shapes.add(sh)
            colors.add(co)
    return sorted(list(shapes)), sorted(list(colors))


def token_embedding(tok: str, shape2id: Dict[str, int], color2id: Dict[str, int]):
    sh, co = split_token(tok)
    vec = np.zeros(len(shape2id) + len(color2id), dtype=np.float32)
    if sh in shape2id:
        vec[shape2id[sh]] = 1.0
    if co in color2id:
        vec[len(shape2id) + color2id[co]] = 1.0
    return vec


def count_color_variety(seq: str) -> int:
    return len(set(split_token(t)[1] for t in seq.strip().split() if len(t) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(split_token(t)[0] for t in seq.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- label processing ----------
labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], dtype=np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], dtype=np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], dtype=np.int64)
num_classes = len(labels)

# ---------- glyph clustering ----------
all_sequences = dsets["train"]["sequence"]
shapes, colors = collect_shapes_colors(all_sequences)
shape2id = {s: i for i, s in enumerate(shapes)}
color2id = {c: i for i, c in enumerate(colors)}

# build embedding matrix for all unique tokens
unique_tokens = sorted(list({tok for seq in all_sequences for tok in seq.split()}))
emb_matrix = np.stack([token_embedding(t, shape2id, color2id) for t in unique_tokens])

best_k, best_score, best_labels = None, -1, None
for k in range(3, 9):  # try 3..8 clusters
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(emb_matrix)
    sil = silhouette_score(emb_matrix, km.labels_) if k > 1 else 0
    if sil > best_score:
        best_k, best_score, best_labels = k, sil, km.labels_
token2cluster = {tok: int(cl) for tok, cl in zip(unique_tokens, best_labels)}

# silhouette per cluster (Si)
samp_sil = (
    silhouette_samples(emb_matrix, best_labels)
    if best_k > 1
    else np.ones(len(best_labels))
)
cluster_sil = {}
for tok, cl, sil_val in zip(unique_tokens, best_labels, samp_sil):
    cluster_sil.setdefault(cl, []).append(float(sil_val))
cluster_sil = {cl: float(np.mean(vals)) for cl, vals in cluster_sil.items()}

experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["cluster_info"] = {
    "k": best_k,
    "overall_silhouette": float(best_score),
    "cluster_sil": cluster_sil,
}


def seq_to_cluster_string(seq: str) -> str:
    return " ".join([f"c{token2cluster.get(tok,0)}" for tok in seq.split()])


train_cluster_str = [seq_to_cluster_string(s) for s in dsets["train"]["sequence"]]
val_cluster_str = [seq_to_cluster_string(s) for s in dsets["dev"]["sequence"]]
test_cluster_str = [seq_to_cluster_string(s) for s in dsets["test"]["sequence"]]

# ---------- vectorizer ----------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
X_train = vectorizer.fit_transform(train_cluster_str).astype(np.float32).toarray()
X_val = vectorizer.transform(val_cluster_str).astype(np.float32).toarray()
X_test = vectorizer.transform(test_cluster_str).astype(np.float32).toarray()


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, n_cls)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(X_train.shape[1], num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- dataloaders ----------
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=128,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=256,
    shuffle=False,
)


# ---------- CCWA computation ----------
def compute_ccwa(seqs: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # assign sequence to dominant cluster
    seq_cluster = []
    for s in seqs:
        clusters = [token2cluster.get(tok, 0) for tok in s.split()]
        # dominant cluster
        seq_cluster.append(max(set(clusters), key=clusters.count))
    # accuracy per cluster
    cluster_correct, cluster_total = {}, {}
    for cl, yt, yp in zip(seq_cluster, y_true, y_pred):
        cluster_total[cl] = cluster_total.get(cl, 0) + 1
        if yt == yp:
            cluster_correct[cl] = cluster_correct.get(cl, 0) + 1
    num = 0.0
    den = 0.0
    for cl in cluster_total:
        Ai = cluster_correct.get(cl, 0) / cluster_total[cl]
        Si = cluster_sil.get(cl, 0.0)
        num += Si * Ai
        den += Si
    return num / den if den > 0 else 0.0


# ---------- training loop ----------
epochs = 8
for epoch in range(1, epochs + 1):
    # train
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

    # validation
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
    preds_arr = np.array(preds, dtype=np.int64)
    acc = (preds_arr == y_val).mean()
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds_arr)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds_arr)
    ccwa = compute_ccwa(dsets["dev"]["sequence"], y_val, preds_arr)

    # log
    experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["losses"]["train"].append(
        train_loss
    )
    experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": ccwa}
    )

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | ACC={acc:.3f} "
        f"CWA={cwa:.3f} SWA={swa:.3f} CCWA={ccwa:.3f}"
    )

# ---------- test evaluation ----------
model.eval()
with torch.no_grad():
    test_preds = model(torch.from_numpy(X_test).to(device)).argmax(1).cpu().numpy()
test_acc = (test_preds == y_test).mean()
test_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
test_ccwa = compute_ccwa(dsets["test"]["sequence"], y_test, test_preds)

experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "ccwa": test_ccwa,
}
experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["predictions"] = test_preds
experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["ground_truth"] = y_test
experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]["sequences"] = dsets["test"][
    "sequence"
]

print(
    f"\nTest  ACC={test_acc:.3f}  CWA={test_cwa:.3f}  SWA={test_swa:.3f}  CCWA={test_ccwa:.3f}"
)

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
