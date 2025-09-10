import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import List

# ---------- directories / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- experiment dict ----------
experiment_data = {
    "RandomClusterAssignment": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ---------- dataset loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for sp in ["train", "dev", "test"]:
        out[sp] = _ld(f"{sp}.csv")
    return out


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    # synthetic fallback
    print("Dataset not found, building synthetic tiny split for demo.")
    shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]

    def synth(n):
        seq = [
            " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            for _ in range(n)
        ]
        labels = np.random.choice(["A", "B", "C"], size=n).tolist()
        return {"sequence": seq, "label": labels}

    dsets = DatasetDict()
    for sp, n in zip(["train", "dev", "test"], [400, 100, 100]):
        dsets[sp] = load_dataset("json", data_files={"train": synth(n)}, split="train")


# ---------- helper metrics ----------
def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ---------- embeddings ----------
def embed_tokens(tokens: List[str]) -> np.ndarray:
    try:
        from transformers import AutoTokenizer, AutoModel

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        mdl = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
        out = []
        with torch.no_grad():
            for t in tokens:
                inp = tok(t, return_tensors="pt").to(device)
                out.append(mdl(**inp).last_hidden_state[:, 0].squeeze(0).cpu().numpy())
        return np.vstack(out)
    except Exception as e:
        print("Falling back to simple embedding:", e)
        vec = []
        for t in tokens:
            a = ord(t[0])
            b = ord(t[1]) if len(t) > 1 else 0
            vec.append([a / 1000.0, b / 1000.0])
        return np.array(vec, dtype=np.float32)


# ---------- build random clusters ----------
train_tokens = [tok for seq in dsets["train"]["sequence"] for tok in seq.split()]
uniq_tokens = sorted(set(train_tokens))
token_embs = embed_tokens(uniq_tokens)

n_clusters = min(8, max(2, len(uniq_tokens) // 3))
print(f"Assigning {len(uniq_tokens)} unique tokens to {n_clusters} random clusters")

# ensure at least one token per cluster (and preferably ≥2 for silhouette)
labels = np.arange(n_clusters).repeat((len(uniq_tokens) // n_clusters) + 1)
np.random.shuffle(labels)
rand_labels = labels[: len(uniq_tokens)]

tok2cluster = {t: int(c) for t, c in zip(uniq_tokens, rand_labels)}

# silhouette values for CCWA (may be low because of randomness)
if n_clusters > 1 and len(set(rand_labels)) > 1 and min(np.bincount(rand_labels)) > 1:
    silh_vals = silhouette_samples(token_embs, rand_labels)
    mean_silh = silhouette_score(token_embs, rand_labels)
else:
    silh_vals, mean_silh = np.ones(len(uniq_tokens)), 1.0
cluster_silh = {
    i: np.mean(silh_vals[np.array(rand_labels) == i]) for i in range(n_clusters)
}
print("Mean silhouette (random clusters):", mean_silh)


# ---------- sequences → cluster token strings ----------
def seq_to_cluster_string(seq):
    return " ".join(f"c{tok2cluster.get(tok, -1)}" for tok in seq.split())


cluster_strings = {
    sp: [seq_to_cluster_string(s) for s in dsets[sp]["sequence"]]
    for sp in ["train", "dev", "test"]
}

# ---------- label to id ----------
labels_all = sorted(set(dsets["train"]["label"]))
lid = {l: i for i, l in enumerate(labels_all)}
y = {
    sp: np.array([lid[l] for l in dsets[sp]["label"]], dtype=np.int64)
    for sp in ["train", "dev", "test"]
}

# ---------- vectorisation ----------
vec_token = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vec_clust = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vec_token.fit(dsets["train"]["sequence"])
vec_clust.fit(cluster_strings["train"])


def build_features(split):
    X1 = vec_token.transform(dsets[split]["sequence"]).toarray().astype(np.float32)
    X2 = vec_clust.transform(cluster_strings[split]).toarray().astype(np.float32)
    return np.hstack([X1, X2])


X = {sp: build_features(sp) for sp in ["train", "dev", "test"]}
print("Feature dimension:", X["train"].shape[1])


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(X["train"].shape[1], len(labels_all)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ---------- dataloaders ----------
def make_loader(split, bs=64):
    ds = TensorDataset(torch.from_numpy(X[split]), torch.from_numpy(y[split]))
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"))


loaders = {sp: make_loader(sp) for sp in ["train", "dev"]}


# ---------- CCWA ----------
def majority_cluster(cs):
    ids = [int(t[1:]) for t in cs.split() if t != "c-1"]
    return max(set(ids), key=ids.count) if ids else -1


def compute_ccwa(split, preds):
    maj = np.array([majority_cluster(s) for s in cluster_strings[split]])
    num = den = 0.0
    for cid in range(n_clusters):
        mask = maj == cid
        if not mask.any():
            continue
        Ai = (preds[mask] == y[split][mask]).mean()
        Si = cluster_silh.get(cid, 0)
        num += Si * Ai
        den += Si
    return num / den if den else 0.0


# ---------- training loop ----------
epochs = 5
for ep in range(1, epochs + 1):
    # train
    model.train()
    tr_loss = 0.0
    for xb, yb in loaders["train"]:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(loaders["train"].dataset)

    # validation
    model.eval()
    vloss, preds = 0.0, []
    with torch.no_grad():
        for xb, yb in loaders["dev"]:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            vloss += criterion(out, yb).item() * xb.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
    vloss /= len(loaders["dev"].dataset)
    preds = np.array(preds)
    acc = (preds == y["dev"]).mean()
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], preds)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], preds)
    ccwa = compute_ccwa("dev", preds)

    # store
    ed = experiment_data["RandomClusterAssignment"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(vloss)
    ed["metrics"]["train"].append({"epoch": ep, "loss": tr_loss})
    ed["metrics"]["val"].append(
        {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": ccwa}
    )
    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={vloss:.4f} "
        f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CCWA={ccwa:.3f}"
    )

# ---------- test evaluation ----------
model.eval()
with torch.no_grad():
    test_pred = model(torch.from_numpy(X["test"]).to(device)).argmax(1).cpu().numpy()
t_acc = (test_pred == y["test"]).mean()
t_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y["test"], test_pred)
t_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y["test"], test_pred)
t_ccwa = compute_ccwa("test", test_pred)
print(f"\nTEST ACC={t_acc:.3f} CWA={t_cwa:.3f} SWA={t_swa:.3f} CCWA={t_ccwa:.3f}")

ed = experiment_data["RandomClusterAssignment"]["SPR_BENCH"]
ed["metrics"]["test"] = {"acc": t_acc, "cwa": t_cwa, "swa": t_swa, "ccwa": t_ccwa}
ed["predictions"] = test_pred
ed["ground_truth"] = y["test"]

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
