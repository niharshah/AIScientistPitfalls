import os, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict, load_dataset

# ------------------------ experiment dict ----------------------------
experiment_data = {
    "batch_size": {
        "SPR_BENCH": {
            "hyperparams": [],  # list of tried batch-sizes
            "metrics": {"train_CompWA": [], "val_CompWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],  # list-of-lists (per batch size)
            "ground_truth": [],  # list-of-lists (per batch size)
        }
    }
}


# --------------------------- utils ----------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # tiny wrapper
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
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def _weighted_acc(weights, y_true, y_pred):
    return sum((w if t == p else 0) for w, t, p in zip(weights, y_true, y_pred)) / max(
        sum(weights), 1
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return _weighted_acc(w, y_true, y_pred)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return _weighted_acc(w, y_true, y_pred)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return _weighted_acc(w, y_true, y_pred)


# ---------------------------- data ----------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
train_sequences, dev_sequences = spr["train"]["sequence"], spr["dev"]["sequence"]


def get_tokens(seqs):
    toks = []
    for s in seqs:
        toks.extend(s.split())
    return toks


all_tokens = get_tokens(train_sequences)
shapes, colors = [t[0] for t in all_tokens], [t[1] for t in all_tokens]
shape_le, color_le = LabelEncoder().fit(shapes), LabelEncoder().fit(colors)
token_vecs = np.stack([shape_le.transform(shapes), color_le.transform(colors)], 1)
k = 8
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(token_vecs)


def sequence_to_histogram(seq: str):
    vec = np.zeros(k, dtype=np.float32)
    for tok in seq.split():
        if len(tok) < 2:
            continue
        lab = kmeans.predict(
            [[shape_le.transform([tok[0]])[0], color_le.transform([tok[1]])[0]]]
        )[0]
        vec[lab] += 1.0
    return vec


X_train = np.stack([sequence_to_histogram(s) for s in train_sequences])
X_dev = np.stack([sequence_to_histogram(s) for s in dev_sequences])
y_train = np.array(spr["train"]["label"], np.float32)
y_dev = np.array(spr["dev"]["label"], np.float32)


# ------------------------ model def ---------------------------------
class SimpleFF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------- hyper-parameter tuning loop --------------------
batch_sizes = [128, 256, 512, 1024]
epochs = 5
for bs in batch_sizes:
    print(f"\n=== Training with batch_size = {bs} ===")
    # data loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=bs)
    # model / optim
    model = SimpleFF(k).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    per_epoch_train_loss, per_epoch_val_loss = [], []
    per_epoch_train_cwa, per_epoch_val_cwa = [], []
    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
        train_loss = run_loss / len(train_loader.dataset)
        # validation
        model.eval()
        val_loss, preds, truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds.extend(
                    (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int).tolist()
                )
                truths.extend(yb.cpu().numpy().astype(int).tolist())
        val_loss /= len(dev_loader.dataset)
        comp_wa = complexity_weighted_accuracy(dev_sequences, truths, preds)
        # record
        per_epoch_train_loss.append(train_loss)
        per_epoch_val_loss.append(val_loss)
        per_epoch_val_cwa.append(comp_wa)
        per_epoch_train_cwa.append(None)  # placeholder
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_CompWA={comp_wa:.4f}")
    # store results for this batch size
    ed = experiment_data["batch_size"]["SPR_BENCH"]
    ed["hyperparams"].append(bs)
    ed["losses"]["train"].append(per_epoch_train_loss)
    ed["losses"]["val"].append(per_epoch_val_loss)
    ed["metrics"]["train_CompWA"].append(per_epoch_train_cwa)
    ed["metrics"]["val_CompWA"].append(per_epoch_val_cwa)
    ed["predictions"].append(preds)
    ed["ground_truth"].append(truths)
    # optional other metrics on dev
    print(
        f"Final Dev CWA={color_weighted_accuracy(dev_sequences, y_dev, preds):.4f}, "
        f"SWA={shape_weighted_accuracy(dev_sequences, y_dev, preds):.4f}"
    )
# ------------------------- save results ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved all experiment data to 'working/experiment_data.npy'")
