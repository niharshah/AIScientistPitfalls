import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict, load_dataset

# ---------------------------- I/O dict --------------------------------
experiment_data = {
    "hidden_dim_sweep": {}  # will be filled with one entry per hidden size
}

# --------------------------- Paths ------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
DATA_PATH = pathlib.Path(
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)  # adapt if needed


# ------------------------ Helper functions ----------------------------
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


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if len(tok) > 0))


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


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum((wi if t == p else 0) for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ----------------------------- Data -----------------------------------
spr = load_spr_bench(DATA_PATH)
train_sequences, dev_sequences = spr["train"]["sequence"], spr["dev"]["sequence"]


def get_tokens(seqs):
    tokens = []
    for s in seqs:
        tokens.extend(s.strip().split())
    return tokens


all_tokens = get_tokens(train_sequences)
shapes = [t[0] for t in all_tokens]
colors = [t[1] for t in all_tokens]
shape_le, color_le = LabelEncoder().fit(shapes), LabelEncoder().fit(colors)
token_vectors = np.stack([shape_le.transform(shapes), color_le.transform(colors)], 1)

k = 8
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(token_vectors)


def sequence_to_histogram(seq: str):
    vec = np.zeros(k, dtype=np.float32)
    for tok in seq.strip().split():
        if len(tok) < 2:
            continue
        s_id = shape_le.transform([tok[0]])[0]
        c_id = color_le.transform([tok[1]])[0]
        label = kmeans.predict([[s_id, c_id]])[0]
        vec[label] += 1.0
    return vec


X_train = np.stack([sequence_to_histogram(s) for s in train_sequences]).astype(
    np.float32
)
X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences]).astype(np.float32)
y_train = np.array(spr["train"]["label"], dtype=np.float32)
y_dev = np.array(spr["dev"]["label"], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------- Model template -----------------------------
class SimpleFF(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------- Hyper-parameter sweep -------------------------
hidden_dims = [16, 32, 64, 128, 256]
epochs, batch_size, lr = 5, 512, 1e-3
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))

for hdim in hidden_dims:
    key = f"SPR_BENCH_h{hdim}"
    experiment_data["hidden_dim_sweep"][key] = {
        "metrics": {"train_CompWA": [], "val_CompWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # loaders (shuffle only for train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    # model/optim
    model = SimpleFF(k, hdim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ---- epochs loop ----
    for ep in range(1, epochs + 1):
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
        train_loss = running_loss / len(train_loader.dataset)
        experiment_data["hidden_dim_sweep"][key]["losses"]["train"].append(train_loss)

        # quick train CompWA (on last mini-batch predictions)
        with torch.no_grad():
            train_preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int).tolist()
            train_truth = yb.cpu().numpy().astype(int).tolist()
            train_comp_wa = complexity_weighted_accuracy(
                train_sequences[-len(train_preds) :], train_truth, train_preds
            )

        # validation
        model.eval()
        val_loss, v_preds, v_truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                v_preds.extend(
                    (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int).tolist()
                )
                v_truths.extend(yb.cpu().numpy().astype(int).tolist())
        val_loss /= len(dev_loader.dataset)
        experiment_data["hidden_dim_sweep"][key]["losses"]["val"].append(val_loss)

        val_comp_wa = complexity_weighted_accuracy(dev_sequences, v_truths, v_preds)
        experiment_data["hidden_dim_sweep"][key]["metrics"]["train_CompWA"].append(
            train_comp_wa
        )
        experiment_data["hidden_dim_sweep"][key]["metrics"]["val_CompWA"].append(
            val_comp_wa
        )

        print(
            f"[hdim {hdim}] Epoch {ep}: train_loss {train_loss:.4f}, "
            f"val_loss {val_loss:.4f}, val_CompWA {val_comp_wa:.4f}"
        )

    # store final preds/labels for this setting
    experiment_data["hidden_dim_sweep"][key]["predictions"] = v_preds
    experiment_data["hidden_dim_sweep"][key]["ground_truth"] = v_truths
    # additional per-setting metrics
    cwa = color_weighted_accuracy(dev_sequences, y_dev, v_preds)
    swa = shape_weighted_accuracy(dev_sequences, y_dev, v_preds)
    print(f"[hdim {hdim}] Final Dev CWA {cwa:.4f}, SWA {swa:.4f}")

# ---------------------------- Save ------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
