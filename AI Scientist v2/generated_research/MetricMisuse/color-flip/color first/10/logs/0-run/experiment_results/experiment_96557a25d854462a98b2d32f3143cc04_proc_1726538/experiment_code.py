import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict, load_dataset

# --------------- experiment data container ---------------------------
experiment_data = {}


# --------------- Utility fns (unchanged) -----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
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
    tokens = []
    for s in seqs:
        tokens.extend(s.strip().split())
    return tokens


all_tokens = get_tokens(train_sequences)
shapes = [t[0] for t in all_tokens]
colors = [t[1] for t in all_tokens]
shape_le, color_le = LabelEncoder().fit(shapes), LabelEncoder().fit(colors)

token_vectors = np.stack(
    [shape_le.transform(shapes), color_le.transform(colors)], axis=1
)

k = 8
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(token_vectors)


def sequence_to_histogram(seq):
    vec = np.zeros(k, np.float32)
    for tok in seq.strip().split():
        if len(tok) < 2:
            continue
        s_id = shape_le.transform([tok[0]])[0]
        c_id = color_le.transform([tok[1]])[0]
        label = kmeans.predict([[s_id, c_id]])[0]
        vec[label] += 1.0
    return vec


X_train = np.stack([sequence_to_histogram(s) for s in train_sequences])
X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences])
y_train = np.asarray(spr["train"]["label"], np.float32)
y_dev = np.asarray(spr["dev"]["label"], np.float32)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------- model definition ------------------------------
class SimpleFF(nn.Module):
    def __init__(self, in_dim: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------- training loop ---------------------------------
def run_experiment(p_drop: float, epochs: int = 5, batch_size: int = 512):
    key = f"dropout_prob_{p_drop}"
    experiment_data[key] = {
        "SPR_BENCH": {
            "metrics": {"train_CompWA": [], "val_CompWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    mdl = SimpleFF(k, p_drop).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(mdl.parameters(), lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)

    for ep in range(1, epochs + 1):
        # ----- train -----
        mdl.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = mdl(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        experiment_data[key]["SPR_BENCH"]["losses"]["train"].append(train_loss)

        # ----- validation -----
        mdl.eval()
        val_loss = 0.0
        preds = []
        truths = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = mdl(xb)
                loss = crit(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds.extend(
                    (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int).tolist()
                )
                truths.extend(yb.cpu().numpy().astype(int).tolist())
        val_loss /= len(dev_loader.dataset)
        cwa_val = complexity_weighted_accuracy(dev_sequences, truths, preds)
        experiment_data[key]["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data[key]["SPR_BENCH"]["metrics"]["val_CompWA"].append(cwa_val)
        experiment_data[key]["SPR_BENCH"]["metrics"]["train_CompWA"].append(None)
        print(
            f"[p={p_drop}] Epoch {ep}: val_loss {val_loss:.4f}, val_CompWA {cwa_val:.4f}"
        )

    # ----- final extra metrics -----
    experiment_data[key]["SPR_BENCH"]["predictions"] = preds
    experiment_data[key]["SPR_BENCH"]["ground_truth"] = truths
    cwa = color_weighted_accuracy(dev_sequences, y_dev, preds)
    swa = shape_weighted_accuracy(dev_sequences, y_dev, preds)
    print(f"[p={p_drop}] Dev CWA {cwa:.4f}, Dev SWA {swa:.4f}")


# --------------------- run sweeps ------------------------------------
for p in [0.0, 0.1, 0.3, 0.5]:
    run_experiment(p)

# --------------------- save results ----------------------------------
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
