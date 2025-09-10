import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict

# ------------------ experiment bookkeeping ---------------------------
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "lrs": [],
            "metrics": {"train_CompWA": [], "val_CompWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------- Utility identical to snippet -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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

train_sequences, dev_sequences = spr["train"]["sequence"], spr["dev"]["sequence"]


def get_tokens(seqs):
    tk = []
    for s in seqs:
        tk.extend(s.strip().split())
    return tk


all_tokens = get_tokens(train_sequences)
shapes, colors = [t[0] for t in all_tokens], [t[1] for t in all_tokens]
shape_le, color_le = LabelEncoder().fit(shapes), LabelEncoder().fit(colors)
token_vectors = np.stack([shape_le.transform(shapes), color_le.transform(colors)], 1)

k = 8
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(token_vectors)


def sequence_to_histogram(seq):
    vec = np.zeros(k, dtype=np.float32)
    for tok in seq.strip().split():
        if len(tok) < 2:
            continue
        s_id, c_id = shape_le.transform([tok[0]])[0], color_le.transform([tok[1]])[0]
        label = kmeans.predict([[s_id, c_id]])[0]
        vec[label] += 1.0
    return vec


X_train = np.stack([sequence_to_histogram(s) for s in train_sequences])
X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences])
y_train = np.array(spr["train"]["label"], dtype=np.float32)
y_dev = np.array(spr["dev"]["label"], dtype=np.float32)

# -------------------------- Torch setup ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class SimpleFF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


batch_size, epochs = 512, 5
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)

lr_grid = [5e-4, 1e-3, 3e-3]

for lr in lr_grid:
    print(f"\n---- Training with learning_rate = {lr} ----")
    experiment_data["learning_rate"]["SPR_BENCH"]["lrs"].append(lr)

    model = SimpleFF(k).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    run_train_losses, run_val_losses = [], []
    run_train_cwa, run_val_cwa = [], []

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        run_train_losses.append(train_loss)

        # validation
        model.eval()
        v_loss, preds, truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                v_loss += loss.item() * xb.size(0)
                preds.extend(
                    (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int).tolist()
                )
                truths.extend(yb.cpu().numpy().astype(int).tolist())
        v_loss /= len(dev_loader.dataset)
        run_val_losses.append(v_loss)

        comp_wa = complexity_weighted_accuracy(dev_sequences, truths, preds)
        run_val_cwa.append(comp_wa)
        run_train_cwa.append(None)  # placeholder for per-epoch train CompWA if desired

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={v_loss:.4f}, val_CompWA={comp_wa:.4f}"
        )

    # aggregate per‐run statistics
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["train"].append(
        run_train_losses
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["val"].append(
        run_val_losses
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["train_CompWA"].append(
        run_train_cwa
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["val_CompWA"].append(
        run_val_cwa
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"].append(truths)

    # optional additional metrics on dev set
    cwa = color_weighted_accuracy(dev_sequences, y_dev, preds)
    swa = shape_weighted_accuracy(dev_sequences, y_dev, preds)
    print(f"Run summary (lr={lr}): Dev CWA={cwa:.4f}, Dev SWA={swa:.4f}")

# ------------------ Save all experiment data -------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
