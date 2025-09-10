# Remove-Cluster-Feature Ablation – single-file runnable script
import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data dict ----------
experiment_data = {
    "RemoveClusterFeat": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _ld(f"{split}.csv")
    return d


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("Dataset not found, generating synthetic toy data")

    def synth(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        seqs = [
            " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            for _ in range(n)
        ]
        labels = np.random.choice(["A", "B", "C"], size=n).tolist()
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [400, 100, 100]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth(n)}, split="train"
        )


# ---------- helpers ----------
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ---------- labels ----------
labels = sorted(set(dsets["train"]["label"]))
lid = {l: i for i, l in enumerate(labels)}
y = {
    sp: np.array([lid[l] for l in dsets[sp]["label"]], dtype=np.int64)
    for sp in ["train", "dev", "test"]
}

# ---------- vectorizer (ONLY raw token n-grams) ----------
vec_token = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vec_token.fit(dsets["train"]["sequence"])


def build_features(split):
    return vec_token.transform(dsets[split]["sequence"]).toarray().astype(np.float32)


X = {sp: build_features(sp) for sp in ["train", "dev", "test"]}
print("Feature dim (X1 only):", X["train"].shape[1])


# ---------- simple MLP ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(X["train"].shape[1], len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ---------- dataloaders ----------
def make_loader(split, bs=64):
    ds = TensorDataset(torch.from_numpy(X[split]), torch.from_numpy(y[split]))
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"))


loaders = {sp: make_loader(sp) for sp in ["train", "dev"]}


# ---------- CCWA surrogate (cluster info removed) ----------
def compute_ccwa(split, preds):
    # Without clusters we fallback to plain accuracy so the metric is still defined.
    return (preds == y[split]).mean()


# ---------- training loop ----------
epochs = 5
for epoch in range(1, epochs + 1):
    # training
    model.train()
    tr_loss = 0.0
    for xb, yb in loaders["train"]:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(loaders["train"].dataset)

    # validation
    model.eval()
    val_loss, val_preds = 0.0, []
    with torch.no_grad():
        for xb, yb in loaders["dev"]:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            val_preds.extend(out.argmax(1).cpu().numpy())
    val_loss /= len(loaders["dev"].dataset)
    val_preds = np.array(val_preds)

    acc = (val_preds == y["dev"]).mean()
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], val_preds)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], val_preds)
    ccwa = compute_ccwa("dev", val_preds)

    # log
    ed = experiment_data["RemoveClusterFeat"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["val"].append(
        {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": ccwa}
    )

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CCWA={ccwa:.3f}"
    )

# ---------- test evaluation ----------
model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(X["test"]).to(device)).argmax(1).cpu().numpy()

test_acc = (preds == y["test"]).mean()
test_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y["test"], preds)
test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y["test"], preds)
test_ccwa = compute_ccwa("test", preds)

print(
    f"\nTEST  ACC={test_acc:.3f} CWA={test_cwa:.3f} SWA={test_swa:.3f} CCWA={test_ccwa:.3f}"
)

ed = experiment_data["RemoveClusterFeat"]["SPR_BENCH"]
ed["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "ccwa": test_ccwa,
}
ed["predictions"] = preds
ed["ground_truth"] = y["test"]

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
