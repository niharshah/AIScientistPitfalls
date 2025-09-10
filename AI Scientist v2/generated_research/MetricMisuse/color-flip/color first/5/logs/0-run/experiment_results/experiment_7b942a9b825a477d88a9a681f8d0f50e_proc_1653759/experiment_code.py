import os, pathlib, torch, numpy as np, sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from datasets import load_dataset, DatasetDict

# --------------- house-keeping -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"val": [], "test": {}},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
}


# --------------- benchmark loader --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for split in ["train", "dev", "test"]:
        dd[split] = _ld(split)
    return dd


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


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


# --------------- load data ---------------------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
if not DATA_PATH.exists():  # fallback tiny synthetic set
    print("Dataset not found, creating synthetic data.")
    shapes, colors = list("ABC"), list("xyz")

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            l = np.random.choice(["L", "M", "N"])
            tok = [
                "%s%s" % (np.random.choice(shapes), np.random.choice(colors))
                for _ in range(np.random.randint(3, 9))
            ]
            seqs.append(" ".join(tok))
            labels.append(l)
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for sp, n in zip(["train", "dev", "test"], [2000, 500, 500]):
        dsets[sp] = load_dataset("json", data_files={"train": synth(n)}, split="train")
else:
    dsets = load_spr_bench(DATA_PATH)

labels = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}


def to_ids(lst):
    return np.array([lab2id[x] for x in lst], dtype=np.int64)


y_train, y_val, y_test = map(
    to_ids, [dsets["train"]["label"], dsets["dev"]["label"], dsets["test"]["label"]]
)
num_classes = len(labels)


# --------------- glyph token processing & clustering ---------------
def token_features(tok):
    # one-hot of shape and color
    shape, color = tok[0], tok[1] if len(tok) > 1 else " "
    return np.array([ord(shape) % 97, ord(color) % 97], dtype=float)  # simple 2-d embed


all_tokens = list({t for seq in dsets["train"]["sequence"] for t in seq.split()})
X_tok = np.stack([token_features(t) for t in all_tokens])
k = min(6, len(all_tokens))  # at least 2 clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(X_tok)
tok2cluster = {tok: int(c) for tok, c in zip(all_tokens, kmeans.labels_)}
sil_samples = silhouette_samples(X_tok, kmeans.labels_)
cluster_silhouette = {
    i: float(np.mean(sil_samples[kmeans.labels_ == i])) for i in range(k)
}


def seq_to_histogram(seq):
    hist = np.zeros(k, dtype=np.float32)
    for tok in seq.split():
        hist[tok2cluster.get(tok, 0)] += 1.0
    return hist


X_train = np.stack([seq_to_histogram(s) for s in dsets["train"]["sequence"]])
X_val = np.stack([seq_to_histogram(s) for s in dsets["dev"]["sequence"]])
X_test = np.stack([seq_to_histogram(s) for s in dsets["test"]["sequence"]])

# --------------- PyTorch dataset --------------
import torch.nn as nn, torch.utils.data as td

train_loader = td.DataLoader(
    td.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=128,
    shuffle=True,
)
val_tensor = (torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(k, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# --------------- CCWA metric ------------------
def ccwa(seqs, y_true, y_pred):
    cluster_acc = {}
    for cid in range(k):
        idx = [
            i
            for i, s in enumerate(seqs)
            if any(tok2cluster.get(t, 0) == cid for t in s.split())
        ]
        if not idx:
            continue
        acc = (y_pred[idx] == y_true[idx]).mean()
        cluster_acc[cid] = acc
    num = sum(cluster_silhouette[c] * cluster_acc.get(c, 0) for c in cluster_silhouette)
    den = sum(cluster_silhouette.values())
    return num / den if den > 0 else 0.0


# --------------- training loop ----------------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_loader.dataset)

    # validation
    model.eval()
    with torch.no_grad():
        out = model(val_tensor[0])
        val_loss = criterion(out, val_tensor[1]).item()
        preds = out.argmax(1).cpu().numpy()
    acc = (preds == y_val).mean()
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
    cc = ccwa(dsets["dev"]["sequence"], y_val, preds)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": cc}
    )

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} validation_loss = {val_loss:.4f} "
        f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CCWA={cc:.3f}"
    )

# --------------- test evaluation --------------
with torch.no_grad():
    test_logits = model(torch.from_numpy(X_test).to(device))
test_preds = test_logits.argmax(1).cpu().numpy()
test_acc = (test_preds == y_test).mean()
test_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
test_ccwa = ccwa(dsets["test"]["sequence"], y_test, test_preds)

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = y_test
experiment_data["SPR_BENCH"]["sequences"] = dsets["test"]["sequence"]
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "ccwa": test_ccwa,
}

print(
    f"Test ACC={test_acc:.3f} CWA={test_cwa:.3f} SWA={test_swa:.3f} CCWA={test_ccwa:.3f}"
)

# --------------- save everything --------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
