import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict, load_dataset

# ---------------------- Reproducibility ------------------------------
torch.manual_seed(0)
np.random.seed(0)
# --------------------------- I/O -------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"weight_decay": {}}


# --------------------------- Data ------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
train_sequences, dev_sequences = spr["train"]["sequence"], spr["dev"]["sequence"]


def get_tokens(seqs):  # flatten tokens
    tokens = []
    for s in seqs:
        tokens.extend(s.strip().split())
    return tokens


all_tokens = get_tokens(train_sequences)
shapes, colors = [t[0] for t in all_tokens], [t[1] for t in all_tokens]
shape_le, color_le = LabelEncoder().fit(shapes), LabelEncoder().fit(colors)
token_vectors = np.stack(
    [shape_le.transform(shapes), color_le.transform(colors)], axis=1
)
k = 8
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(token_vectors)


def sequence_to_histogram(seq):
    vec = np.zeros(k, dtype=np.float32)
    for tok in seq.strip().split():
        if len(tok) < 2:
            continue
        s_id, c_id = shape_le.transform([tok[0]])[0], color_le.transform([tok[1]])[0]
        vec[kmeans.predict([[s_id, c_id]])[0]] += 1.0
    return vec


X_train = np.stack([sequence_to_histogram(s) for s in train_sequences])
X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences])
y_train = np.array(spr["train"]["label"], dtype=np.float32)
y_dev = np.array(spr["dev"]["label"], dtype=np.float32)


# ----------------------- Metrics helpers -----------------------------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------------------------- Model ----------------------------------
class SimpleFF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------- Hyperparameter sweep --------------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
batch_size, epochs, lr = 512, 5, 1e-3
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
for wd in weight_decays:
    run_key = f"wd_{wd}"
    experiment_data["weight_decay"][run_key] = {
        "metrics": {"train_CompWA": [], "val_CompWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = SimpleFF(k).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(1, epochs + 1):
        # -------- training ----------
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
        # quick train CompWA
        with torch.no_grad():
            tr_logits = model(torch.from_numpy(X_train).to(device))
            tr_preds = (torch.sigmoid(tr_logits) > 0.5).cpu().numpy().astype(int)
            tr_comp = complexity_weighted_accuracy(
                train_sequences, y_train.astype(int), tr_preds
            )
        # -------- validation ---------
        model.eval()
        val_loss, preds, truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                preds.extend(
                    (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int).tolist()
                )
                truths.extend(yb.cpu().numpy().astype(int).tolist())
        val_loss /= len(dev_loader.dataset)
        val_comp = complexity_weighted_accuracy(dev_sequences, truths, preds)
        # -------- logging ------------
        exp = experiment_data["weight_decay"][run_key]
        exp["losses"]["train"].append(train_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train_CompWA"].append(tr_comp)
        exp["metrics"]["val_CompWA"].append(val_comp)
        if epoch == epochs:  # store final predictions only once
            exp["predictions"] = preds
            exp["ground_truth"] = truths
        print(
            f"[wd={wd}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_CompWA={val_comp:.4f}"
        )
    # per-run extra metrics
    exp["CWA"] = color_weighted_accuracy(dev_sequences, y_dev.astype(int), preds)
    exp["SWA"] = shape_weighted_accuracy(dev_sequences, y_dev.astype(int), preds)
# ------------------------- Save to disk ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
