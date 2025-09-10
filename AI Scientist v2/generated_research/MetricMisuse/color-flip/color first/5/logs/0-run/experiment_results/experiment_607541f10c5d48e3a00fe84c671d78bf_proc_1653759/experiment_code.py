import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from datasets import load_dataset, DatasetDict

# --------------------- setup & GPU handling ---------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- experiment data container ---------------------------------
experiment_data = {
    "SPR_BENCH_cluster_hist": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "silhouette": {},
    }
}


# --------------------- data utilities --------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def CWA(seqs, y_true, y_pred):
    w = [color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def SWA(seqs, y_true, y_pred):
    w = [shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# --------------------- load dataset ----------------------------------------------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
dsets = load_spr_bench(DATA_PATH)

# --------------------- glyph embedding & clustering ------------------------------
all_tokens = set(tok for seq in dsets["train"]["sequence"] for tok in seq.split())
shapes = sorted(set(t[0] for t in all_tokens))
colors = sorted(set(t[1] for t in all_tokens if len(t) > 1))
shape2idx = {s: i for i, s in enumerate(shapes)}
color2idx = {c: i for i, c in enumerate(colors)}


def embed_token(tok: str):
    s_vec = np.eye(len(shapes))[shape2idx[tok[0]]]
    c_vec = np.eye(len(colors))[color2idx[tok[1]]]
    return np.concatenate([s_vec, c_vec])


token_vecs = np.stack([embed_token(t) for t in all_tokens])
K = min(8, len(all_tokens))
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0).fit(token_vecs)
token2cluster = {tok: int(cl) for tok, cl in zip(all_tokens, kmeans.labels_)}

# ---- FIX: shift silhouette scores to strictly-positive [0,1] range --------------
raw_sil = (
    silhouette_samples(token_vecs, kmeans.labels_)
    if K > 1
    else np.ones(len(all_tokens))
)
shifted_sil = (raw_sil + 1) / 2  # now in [0,1]
cluster_sil = {i: shifted_sil[kmeans.labels_ == i].mean() for i in range(K)}
experiment_data["SPR_BENCH_cluster_hist"]["silhouette"] = cluster_sil


# --------------------- sequence → histogram --------------------------------------
def seq_to_hist(seq: str) -> np.ndarray:
    hist = np.zeros(K, dtype=np.float32)
    for tok in seq.split():
        hist[token2cluster[tok]] += 1.0
    return hist / max(len(seq.split()), 1)


def build_split(name):
    X = np.stack([seq_to_hist(s) for s in dsets[name]["sequence"]]).astype(np.float32)
    lbl_set = sorted(set(dsets["train"]["label"]))
    y = np.array([lbl_set.index(l) for l in dsets[name]["label"]], dtype=np.int64)
    return X, y, lbl_set


X_tr, y_tr, labels = build_split("train")
X_val, y_val, _ = build_split("dev")
X_te, y_te, _ = build_split("test")
num_classes = len(labels)

# --------------------- dataloaders -----------------------------------------------
batch_sz = 64
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
    batch_size=batch_sz,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_sz
)


# --------------------- model ------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(K, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --------------------- CCWA metric using shifted silhouette ----------------------
def CCWA(seqs, y_true, y_pred):
    cluster_correct = {i: [0, 0] for i in range(K)}  # [correct, total]
    for s, yt, yp in zip(seqs, y_true, y_pred):
        involved = {token2cluster[tok] for tok in s.split()}
        for c in involved:
            cluster_correct[c][1] += 1
            if yt == yp:
                cluster_correct[c][0] += 1
    num = sum(
        cluster_sil[i]
        * (
            cluster_correct[i][0] / cluster_correct[i][1]
            if cluster_correct[i][1] > 0
            else 0
        )
        for i in range(K)
    )
    den = sum(cluster_sil.values())
    return num / den if den > 0 else 0.0


# --------------------- training loop ---------------------------------------------
best_val_acc, best_state = -1.0, None
epochs = 10
for epoch in range(1, epochs + 1):
    # --- train ---
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

    # --- validation ---
    model.eval()
    val_loss, preds = 0.0, []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_loss += criterion(out, yb).item() * xb.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
    val_loss /= len(val_loader.dataset)
    preds = np.array(preds)

    acc = (preds == y_val).mean()
    cwa = CWA(dsets["dev"]["sequence"], y_val, preds)
    swa = SWA(dsets["dev"]["sequence"], y_val, preds)
    ccwa = CCWA(dsets["dev"]["sequence"], y_val, preds)

    experiment_data["SPR_BENCH_cluster_hist"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH_cluster_hist"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH_cluster_hist"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": ccwa}
    )

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | ACC={acc:.3f} "
        f"CWA={cwa:.3f} SWA={swa:.3f} CCWA={ccwa:.3f}"
    )

    if acc > best_val_acc:
        best_val_acc = acc
        best_state = model.state_dict()

# --------------------- test -------------------------------------------------------
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(X_te).to(device))
    test_pred = logits.argmax(1).cpu().numpy()

test_acc = (test_pred == y_te).mean()
test_cwa = CWA(dsets["test"]["sequence"], y_te, test_pred)
test_swa = SWA(dsets["test"]["sequence"], y_te, test_pred)
test_ccwa = CCWA(dsets["test"]["sequence"], y_te, test_pred)

experiment_data["SPR_BENCH_cluster_hist"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "ccwa": test_ccwa,
}
experiment_data["SPR_BENCH_cluster_hist"]["predictions"] = test_pred
experiment_data["SPR_BENCH_cluster_hist"]["ground_truth"] = y_te

print(
    f"\nTEST: ACC={test_acc:.3f} CWA={test_cwa:.3f} SWA={test_swa:.3f} CCWA={test_ccwa:.3f}"
)

# --------------------- persist results -------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
