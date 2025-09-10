import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import List, Dict

# --------- working dir & experiment dict ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "BinaryCountAblation": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------- load SPR_BENCH or synthetic tiny version ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):  # helper
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _ld(f"{sp}.csv")
    return d


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:  # tiny synthetic fallback
    print("Dataset not found – using synthetic tiny split for demo")

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
    for sp, n in zip(["train", "dev", "test"], [400, 100, 100]):
        dsets[sp] = load_dataset("json", data_files={"train": synth(n)}, split="train")


# --------- misc helpers ----------
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


# --------- token embedding (BERT if available else ord) ----------
def embed_tokens(tokens: List[str]) -> np.ndarray:
    try:
        from transformers import AutoTokenizer, AutoModel

        tok, mdl = (
            AutoTokenizer.from_pretrained("bert-base-uncased"),
            AutoModel.from_pretrained("bert-base-uncased").to(device).eval(),
        )
        with torch.no_grad():
            return np.vstack(
                [
                    mdl(**tok(t, return_tensors="pt").to(device))
                    .last_hidden_state[:, 0]
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    for t in tokens
                ]
            )
    except Exception as e:
        print("BERT unavailable, using ord embedding:", e)
        return np.array(
            [
                [ord(t[0]) / 1000, (ord(t[1]) if len(t) > 1 else 0) / 1000]
                for t in tokens
            ],
            dtype=np.float32,
        )


# --------- build token clusters ----------
train_tokens = [tok for seq in dsets["train"]["sequence"] for tok in seq.split()]
uniq_tokens = sorted(set(train_tokens))
token_embs = embed_tokens(uniq_tokens)
n_clusters = min(8, max(2, len(uniq_tokens) // 3))
print(f"Clustering {len(uniq_tokens)} tokens into {n_clusters} clusters")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(token_embs)
tok2cluster = {t: int(c) for t, c in zip(uniq_tokens, kmeans.labels_)}
silh_vals = (
    silhouette_samples(token_embs, kmeans.labels_)
    if n_clusters > 1
    else np.ones(len(uniq_tokens))
)
cluster_silh = {i: silh_vals[kmeans.labels_ == i].mean() for i in range(n_clusters)}


def seq_to_cluster_string(seq: str) -> str:
    return " ".join(f"c{tok2cluster.get(tok,-1)}" for tok in seq.split())


cluster_strings = {
    sp: [seq_to_cluster_string(s) for s in dsets[sp]["sequence"]]
    for sp in ["train", "dev", "test"]
}

# --------- label encoding ----------
labels = sorted(set(dsets["train"]["label"]))
lid = {l: i for i, l in enumerate(labels)}
y = {
    sp: np.array([lid[l] for l in dsets[sp]["label"]], dtype=np.int64)
    for sp in ["train", "dev", "test"]
}

# --------- vectorisers (keep vocabulary) ----------
vec_token = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vec_clust = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
vec_token.fit(dsets["train"]["sequence"])
vec_clust.fit(cluster_strings["train"])


# --------- Binary Count Ablation: convert counts>0 to 1 ----------
def build_features(split: str) -> np.ndarray:
    X1 = (
        (vec_token.transform(dsets[split]["sequence"]) > 0).astype(np.float32).toarray()
    )
    X2 = (vec_clust.transform(cluster_strings[split]) > 0).astype(np.float32).toarray()
    return np.hstack([X1, X2])


X = {sp: build_features(sp) for sp in ["train", "dev", "test"]}
print("Binary feature dim:", X["train"].shape[1])


# --------- MLP ----------
class MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 256), nn.ReLU(), nn.Linear(256, out))

    def forward(self, x):
        return self.net(x)


model = MLP(X["train"].shape[1], len(labels)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# --------- dataloaders ----------
def make_loader(split, bs=64):
    ds = TensorDataset(torch.from_numpy(X[split]), torch.from_numpy(y[split]))
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"))


loaders = {sp: make_loader(sp) for sp in ["train", "dev"]}


# --------- CCWA ----------
def majority_cluster(seq_clusters: str) -> int:
    ids = [int(t[1:]) for t in seq_clusters.split() if t != "c-1"]
    return max(set(ids), key=ids.count) if ids else -1


def compute_ccwa(split: str, preds: np.ndarray) -> float:
    maj = np.array([majority_cluster(cs) for cs in cluster_strings[split]])
    num = den = 0.0
    for cid in range(n_clusters):
        m = maj == cid
        if not m.any():
            continue
        Ai = (preds[m] == y[split][m]).mean()
        Si = cluster_silh.get(cid, 0)
        num += Si * Ai
        den += Si
    return num / den if den else 0.0


# --------- training loop ----------
epochs = 5
for ep in range(1, epochs + 1):
    # train
    model.train()
    tr_loss = 0.0
    for xb, yb in loaders["train"]:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optim.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(loaders["train"].dataset)
    # val
    model.eval()
    v_loss = []
    v_preds = []
    with torch.no_grad():
        for xb, yb in loaders["dev"]:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            v_loss.append(loss.item() * xb.size(0))
            v_preds.extend(out.argmax(1).cpu().numpy())
    v_loss = sum(v_loss) / len(loaders["dev"].dataset)
    v_preds = np.array(v_preds)
    acc = (v_preds == y["dev"]).mean()
    cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], v_preds)
    swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y["dev"], v_preds)
    ccwa = compute_ccwa("dev", v_preds)

    ed = experiment_data["BinaryCountAblation"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(v_loss)
    ed["metrics"]["val"].append(
        {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "ccwa": ccwa}
    )

    print(
        f"E{ep}: train_loss={tr_loss:.4f} val_loss={v_loss:.4f} ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CCWA={ccwa:.3f}"
    )

# --------- test evaluation ----------
model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(X["test"]).to(device)).argmax(1).cpu().numpy()
t_acc = (preds == y["test"]).mean()
t_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y["test"], preds)
t_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y["test"], preds)
t_ccwa = compute_ccwa("test", preds)
print(f"\nTEST ACC={t_acc:.3f} CWA={t_cwa:.3f} SWA={t_swa:.3f} CCWA={t_ccwa:.3f}")

ed = experiment_data["BinaryCountAblation"]["SPR_BENCH"]
ed["metrics"]["test"] = {"acc": t_acc, "cwa": t_cwa, "swa": t_swa, "ccwa": t_ccwa}
ed["predictions"] = preds
ed["ground_truth"] = y["test"]

# --------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
