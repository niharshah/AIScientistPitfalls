import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict

# -------------------------------------------------- misc / IO
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {}

# -------------------------------------------------- GPU / device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------- dataset helpers
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split()))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# -------------------------------------------------- data path
DATA_ENV = os.getenv("SPR_DIR", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
DATA_PATH = pathlib.Path(DATA_ENV)
spr = load_spr_bench(DATA_PATH)

train_seqs = spr["train"]["sequence"]
dev_seqs = spr["dev"]["sequence"]
test_seqs = spr["test"]["sequence"]
y_train = torch.tensor(spr["train"]["label"]).float()
y_dev = torch.tensor(spr["dev"]["label"]).float()
y_test = torch.tensor(spr["test"]["label"]).float()


# -------------------------------------------------- build global encoders for glyph parts
def all_tokens(seqs):
    toks = []
    for s in seqs:
        toks.extend(s.split())
    return toks


tokens = all_tokens(train_seqs)
shape_le = LabelEncoder().fit([t[0] for t in tokens])
color_le = LabelEncoder().fit([t[1] for t in tokens])
n_shapes = len(shape_le.classes_)
n_colors = len(color_le.classes_)

# pre-compute 2-D numerical embedding per token for clustering
token_embed = np.stack(
    [
        shape_le.transform([t[0] for t in tokens]),
        color_le.transform([t[1] for t in tokens]),
    ],
    1,
)


# -------------------------------------------------- feature builders
def seq_to_histograms(seq: str, kmeans, k):
    cluster_hist = np.zeros(k, dtype=np.float32)
    shape_hist = np.zeros(n_shapes, dtype=np.float32)
    color_hist = np.zeros(n_colors, dtype=np.float32)
    for tok in seq.split():
        sid = shape_le.transform([tok[0]])[0]
        cid = color_le.transform([tok[1]])[0]
        lbl = kmeans.predict([[sid, cid]])[0]
        cluster_hist[lbl] += 1
        shape_hist[sid] += 1
        color_hist[cid] += 1
    return np.concatenate([cluster_hist, shape_hist, color_hist])


# -------------------------------------------------- model
class FFNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------------------------------------- experiment loop
for k in [8, 16, 32]:
    print(f"\n========== k = {k} ==========")
    experiment_data[f"k={k}"] = {
        "metrics": {"val": {"CWA": [], "SWA": [], "CompWA": []}},
        "losses": {"train": [], "val": []},
    }
    # clustering
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048, n_init=10)
    kmeans.fit(token_embed)

    # build matrices
    X_train = np.stack([seq_to_histograms(s, kmeans, k) for s in train_seqs])
    X_dev = np.stack([seq_to_histograms(s, kmeans, k) for s in dev_seqs])

    train_ds = TensorDataset(torch.tensor(X_train).float(), y_train)
    dev_ds = TensorDataset(torch.tensor(X_dev).float(), y_dev)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=512)

    model = FFNet(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 12
    for epoch in range(1, EPOCHS + 1):
        # ---- train
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        experiment_data[f"k={k}"]["losses"]["train"].append(train_loss)

        # ---- validate
        model.eval()
        vloss, preds, truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
                truths.extend(yb.cpu().numpy().astype(int))
        vloss /= len(dev_loader.dataset)
        experiment_data[f"k={k}"]["losses"]["val"].append(vloss)

        cwa = color_weighted_accuracy(dev_seqs, truths, preds)
        swa = shape_weighted_accuracy(dev_seqs, truths, preds)
        comp = complexity_weighted_accuracy(dev_seqs, truths, preds)
        experiment_data[f"k={k}"]["metrics"]["val"]["CWA"].append(cwa)
        experiment_data[f"k={k}"]["metrics"]["val"]["SWA"].append(swa)
        experiment_data[f"k={k}"]["metrics"]["val"]["CompWA"].append(comp)

        print(
            f"Epoch {epoch}: val_loss={vloss:.4f} | CWA={cwa:.4f} SWA={swa:.4f} CompWA={comp:.4f}"
        )

# -------------------------------------------------- save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy to ./working")
