import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict

# ------------------------- housekeeping -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- dataset --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# try environment variable fall-back
root_path = pathlib.Path(
    os.environ.get("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(root_path)
train_seqs, dev_seqs = spr["train"]["sequence"], spr["dev"]["sequence"]
y_train = np.array(spr["train"]["label"], dtype=np.float32)
y_dev = np.array(spr["dev"]["label"], dtype=np.float32)


# --------------------- helper functions -----------------------------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


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


def complexity_weighted_accuracy2(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# -------------------- glyph vectorisation ---------------------------
def all_tokens(seqs):
    toks = []
    for s in seqs:
        toks.extend(s.strip().split())
    return toks


tokens = all_tokens(train_seqs)
shape_le = LabelEncoder().fit([t[0] for t in tokens])
color_le = LabelEncoder().fit([t[1] for t in tokens])

token_vecs = np.stack(
    [
        shape_le.transform([t[0] for t in tokens]),
        color_le.transform([t[1] for t in tokens]),
    ],
    axis=1,
)

# -------------- choose best k via silhouette on sample --------------
candidate_k = [4, 8, 16, 32]
sample_idx = np.random.choice(
    len(token_vecs), size=min(4000, len(token_vecs)), replace=False
)
sample_vecs = token_vecs[sample_idx]

best_k, best_score = None, -1
for k in candidate_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sample_vecs)
    score = silhouette_score(sample_vecs, km.labels_)
    print(f"k={k:2d} silhouette={score:.3f}")
    if score > best_score:
        best_k, best_score = k, score
print(f"Chosen k = {best_k} with silhouette {best_score:.3f}")

# ----------------- final clustering with chosen k -------------------
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(token_vecs)


def seq_to_hist(seq: str, k: int):
    h = np.zeros(k, dtype=np.float32)
    for tok in seq.strip().split():
        if len(tok) < 2:
            continue
        s_id = shape_le.transform([tok[0]])[0]
        c_id = color_le.transform([tok[1]])[0]
        cl = kmeans.predict([[s_id, c_id]])[0]
        h[cl] += 1.0
    return h


X_train = np.stack([seq_to_hist(s, best_k) for s in train_seqs])
X_dev = np.stack([seq_to_hist(s, best_k) for s in dev_seqs])

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512)


# ------------------------- model ------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = MLP(best_k).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------- training / validation loop ---------------------
best_val_loss, patience, wait = 1e9, 2, 0
for epoch in range(1, 7):  # 6 epochs max
    # ----- train -----
    model.train()
    run_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * xb.size(0)
    tr_loss = run_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # ----- validate -----
    model.eval()
    v_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            v_loss += loss.item() * xb.size(0)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            gts.extend(yb.cpu().numpy().astype(int))
    v_loss /= len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)

    cwa2 = complexity_weighted_accuracy2(dev_seqs, gts, preds)
    cwa = color_weighted_accuracy(dev_seqs, gts, preds)
    swa = shape_weighted_accuracy(dev_seqs, gts, preds)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"CWA2": cwa2, "CWA": cwa, "SWA": swa}
    )
    print(
        f"Epoch {epoch}: val_loss={v_loss:.4f} | CWA2={cwa2:.4f} | CWA={cwa:.4f} | SWA={swa:.4f}"
    )

    # early stopping
    if v_loss < best_val_loss - 1e-4:
        best_val_loss = v_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping...")
            break

# ------------------ final bookkeeping --------------------------------
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nExperiment finished. Data saved to {working_dir}/experiment_data.npy")
