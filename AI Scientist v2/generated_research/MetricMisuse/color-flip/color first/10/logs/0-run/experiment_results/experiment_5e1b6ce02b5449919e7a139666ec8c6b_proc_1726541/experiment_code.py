import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict

# -----------------------  set-up & bookkeeping  ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"num_clusters_k": {"SPR_BENCH": {}}}


# --------------------------- utilities ------------------------------
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


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


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


# ---------------------------- data ----------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
train_sequences, dev_sequences = spr["train"]["sequence"], spr["dev"]["sequence"]


# label encoders for glyph parts (keep fixed across k)
def get_tokens(seqs):
    toks = []
    for s in seqs:
        toks.extend(s.strip().split())
    return toks


all_tokens = get_tokens(train_sequences)
shape_le = LabelEncoder().fit([t[0] for t in all_tokens])
color_le = LabelEncoder().fit([t[1] for t in all_tokens])

token_vectors = np.stack(
    [
        shape_le.transform([t[0] for t in all_tokens]),
        color_le.transform([t[1] for t in all_tokens]),
    ],
    axis=1,
)

y_train = np.array(spr["train"]["label"], dtype=np.float32)
y_dev = np.array(spr["dev"]["label"], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------- training function per k ----------------------
def run_experiment(k: int, epochs: int = 5, batch_size: int = 512):
    print(f"\n===== Training with k = {k} clusters =====")
    # --- clustering ---
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(token_vectors)

    def sequence_to_histogram(seq: str):
        vec = np.zeros(k, dtype=np.float32)
        for tok in seq.strip().split():
            if len(tok) < 2:  # skip malformed
                continue
            s_id = shape_le.transform([tok[0]])[0]
            c_id = color_le.transform([tok[1]])[0]
            label = kmeans.predict([[s_id, c_id]])[0]
            vec[label] += 1.0
        return vec

    X_train = np.stack([sequence_to_histogram(s) for s in train_sequences])
    X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences])

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)

    # --- model ---
    class SimpleFF(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = SimpleFF(k).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # data containers
    k_dict = {
        "metrics": {"train_CompWA": [], "val_CompWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        k_dict["losses"]["train"].append(train_loss)

        # ---- validation ----
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
        k_dict["losses"]["val"].append(vloss)

        comp_wa = complexity_weighted_accuracy(dev_sequences, truths, preds)
        k_dict["metrics"]["train_CompWA"].append(None)  # skipped for brevity
        k_dict["metrics"]["val_CompWA"].append(comp_wa)
        print(f"Epoch {epoch}: val_loss={vloss:.4f}, val_CompWA={comp_wa:.4f}")

    # ---- final predictions / metrics ----
    k_dict["predictions"] = preds
    k_dict["ground_truth"] = truths
    cwa = color_weighted_accuracy(dev_sequences, truths, preds)
    swa = shape_weighted_accuracy(dev_sequences, truths, preds)
    print(f"k={k} -> Dev CWA: {cwa:.4f}, Dev SWA: {swa:.4f}")
    return k_dict


# ------------------------ hyper-parameter loop ----------------------
for k_val in [4, 8, 16, 32]:
    experiment_data["num_clusters_k"]["SPR_BENCH"][f"k={k_val}"] = run_experiment(k_val)

# --------------------------- save -----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll experiments finished and saved to 'experiment_data.npy'.")
