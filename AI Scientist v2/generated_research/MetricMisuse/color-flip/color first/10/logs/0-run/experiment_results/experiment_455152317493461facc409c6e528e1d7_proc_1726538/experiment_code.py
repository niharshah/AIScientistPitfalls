import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict, load_dataset

# ------------- where to save everything ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------ Experiment dict -----
experiment_data = {
    "num_epochs": {  # hyper-parameter being tuned
        "SPR_BENCH": {
            "config_values": [],  # list of epochs tried
            "losses": {"train": [], "val": []},  # list-of-lists
            "metrics": {"train_CompWA": [], "val_CompWA": []},
            "predictions": [],  # list of arrays
            "ground_truth": [],  # list of arrays
        }
    }
}


# ---------------- Utility functions (unchanged) ----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


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


# ----------------------------- Data ----------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

train_sequences = spr["train"]["sequence"]
dev_sequences = spr["dev"]["sequence"]


# tokenise
def get_tokens(seqs):
    toks = []
    for s in seqs:
        toks.extend(s.strip().split())
    return toks


all_tokens = get_tokens(train_sequences)
shapes = [t[0] for t in all_tokens]
colors = [t[1] for t in all_tokens]
shape_le = LabelEncoder().fit(shapes)
color_le = LabelEncoder().fit(colors)
token_vectors = np.stack(
    [shape_le.transform(shapes), color_le.transform(colors)], axis=1
)

# clustering
k = 8
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(token_vectors)


def sequence_to_histogram(seq):
    vec = np.zeros(k, dtype=np.float32)
    for tok in seq.strip().split():
        if len(tok) < 2:
            continue
        sid = shape_le.transform([tok[0]])[0]
        cid = color_le.transform([tok[1]])[0]
        label = kmeans.predict([[sid, cid]])[0]
        vec[label] += 1.0
    return vec


X_train = np.stack([sequence_to_histogram(s) for s in train_sequences])
X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences])
y_train = np.array(spr["train"]["label"], dtype=np.float32)
y_dev = np.array(spr["dev"]["label"], dtype=np.float32)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- Model definition -----------------------------
class SimpleFF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------- Hyper-parameter sweep ------------------------------
epoch_options = [5, 10, 20, 30, 40, 50]
batch_size = 512
patience = 5  # early stopping
for max_epochs in epoch_options:
    print(f"\n=== Training with max_epochs = {max_epochs} ===")
    experiment_data["num_epochs"]["SPR_BENCH"]["config_values"].append(max_epochs)

    model = SimpleFF(k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)

    best_val_loss = float("inf")
    best_pred, best_truth = None, None
    best_train_losses, best_val_losses = [], []
    best_train_cwa, best_val_cwa = [], []
    epochs_without_improve = 0

    per_epoch_train_loss, per_epoch_val_loss = [], []
    per_epoch_train_compwa, per_epoch_val_compwa = [], []

    for epoch in range(1, max_epochs + 1):
        # --- train ---
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

        # --- validation ---
        model.eval()
        val_loss, preds, truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds.extend((torch.sigmoid(out) > 0.5).cpu().numpy().astype(int))
                truths.extend(yb.cpu().numpy().astype(int))
        val_loss /= len(dev_loader.dataset)

        comp_wa = complexity_weighted_accuracy(dev_sequences, truths, preds)
        per_epoch_train_loss.append(train_loss)
        per_epoch_val_loss.append(val_loss)
        per_epoch_val_compwa.append(comp_wa)
        per_epoch_train_compwa.append(None)  # not evaluated

        print(
            f"Epoch {epoch}/{max_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_CompWA={comp_wa:.4f}"
        )

        # early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_pred, best_truth = preds.copy(), truths.copy()
            best_train_losses = per_epoch_train_loss.copy()
            best_val_losses = per_epoch_val_loss.copy()
            best_train_cwa = per_epoch_train_compwa.copy()
            best_val_cwa = per_epoch_val_compwa.copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ------------- record best run -----------------------------------
    experiment_data["num_epochs"]["SPR_BENCH"]["losses"]["train"].append(
        best_train_losses
    )
    experiment_data["num_epochs"]["SPR_BENCH"]["losses"]["val"].append(best_val_losses)
    experiment_data["num_epochs"]["SPR_BENCH"]["metrics"]["train_CompWA"].append(
        best_train_cwa
    )
    experiment_data["num_epochs"]["SPR_BENCH"]["metrics"]["val_CompWA"].append(
        best_val_cwa
    )
    experiment_data["num_epochs"]["SPR_BENCH"]["predictions"].append(best_pred)
    experiment_data["num_epochs"]["SPR_BENCH"]["ground_truth"].append(best_truth)

    # optional auxiliary metrics
    cwa = color_weighted_accuracy(dev_sequences, best_truth, best_pred)
    swa = shape_weighted_accuracy(dev_sequences, best_truth, best_pred)
    print(f"Final Dev CWA: {cwa:.4f}, Dev SWA: {swa:.4f}")

# ---------------------- persist everything ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
