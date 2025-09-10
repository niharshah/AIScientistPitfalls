import os, math, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------- DATA ------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y, yhat):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, yhat) if yt == yp) / max(1, sum(w))


def swa(seqs, y, yhat):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, yhat) if yt == yp) / max(1, sum(w))


def harmonic(a, b):
    return 2 * a * b / (a + b) if (a + b) else 0.0


# ------------------------------------------------------ CLUSTERING ---------------------------------------------------
def glyph_features(tok: str):
    code = [ord(c) for c in tok]
    first, mean_rest = code[0], (
        (sum(code[1:]) / (len(code) - 1)) if len(code) > 1 else 0.0
    )
    return [first, mean_rest]


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
all_tokens = sorted({t for s in spr["train"]["sequence"] for t in s.split()})
X = np.array([glyph_features(t) for t in all_tokens])
n_clusters = max(8, int(math.sqrt(len(all_tokens))))
print(f"Clustering {len(all_tokens)} glyphs into {n_clusters} clusters…")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
glyph2cluster = {
    tok: int(c) + 1 for tok, c in zip(all_tokens, kmeans.labels_)
}  # +1 because 0 will be PAD

# ------------------------------------------------------ DATASET ------------------------------------------------------
PAD = 0


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lbl = split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        ids = [
            glyph2cluster.get(t, 1) for t in self.seq[idx].split()
        ]  # OOV -> cluster 1
        return {
            "input": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(len(ids), dtype=torch.long),
            "label": torch.tensor(int(self.lbl[idx]), dtype=torch.long),
            "raw": self.seq[idx],
        }


def collate(batch):
    maxlen = max(b["len"] for b in batch)
    padded = [
        torch.cat([b["input"], torch.zeros(maxlen - b["len"], dtype=torch.long)])
        for b in batch
    ]
    return {
        "input": torch.stack(padded),
        "len": torch.stack([b["len"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


train_dl = DataLoader(
    SPRDataset(spr["train"]), batch_size=256, shuffle=True, collate_fn=collate
)
dev_dl = DataLoader(
    SPRDataset(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate
)
test_dl = DataLoader(
    SPRDataset(spr["test"]), batch_size=512, shuffle=False, collate_fn=collate
)

num_labels = len(set(spr["train"]["label"]))
vocab_size = n_clusters + 1  # plus PAD


# ------------------------------------------------------ MODEL --------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid, nclass):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True)
        self.lin = nn.Linear(hid, nclass)

    def forward(self, x, lens):
        x, lens = x.to(device), lens.to(device)
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        return self.lin(h.squeeze(0))


# ------------------------------------------------ CLUSTER-NORMALISED ACCURACY ----------------------------------------
def cna(seqs, y_true, y_pred):
    # map each sequence to set of clusters present
    per_cluster = {c: [0, 0] for c in range(1, n_clusters + 1)}
    for s, yt, yp in zip(seqs, y_true, y_pred):
        clusters = {glyph2cluster.get(t, 1) for t in s.split()}
        for c in clusters:
            per_cluster[c][1] += 1
            if yt == yp:
                per_cluster[c][0] += 1
    accs = [(c_ok / (c_tot)) if c_tot else 0.0 for c_ok, c_tot in per_cluster.values()]
    return sum(accs) / len(accs)


# ------------------------------------------------ EXPERIMENT DATA ----------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------ TRAIN / EVAL -------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    ys, yh, raws = [], [], []
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch_t["input"], batch_t["len"])
            loss_sum += crit(out, batch_t["label"]).item() * batch_t["label"].size(0)
            ys.extend(batch_t["label"].cpu().tolist())
            yh.extend(out.argmax(-1).cpu().tolist())
            raws.extend(batch["raw"])
    loss = loss_sum / len(loader.dataset)
    _cwa, _swa = cwa(raws, ys, yh), swa(raws, ys, yh)
    _cna = cna(raws, ys, yh)
    return loss, _cwa, _swa, harmonic(_cwa, _swa), _cna, ys, yh, raws


def train(lr=2e-3, epochs=5):
    print(f"Training with lr={lr}")
    model = GRUClassifier(vocab_size, 32, 64, num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0
        tr_loss = 0.0
        for batch in train_dl:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            out = model(batch_t["input"], batch_t["len"])
            loss = crit(out, batch_t["label"])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch_t["label"].size(0)
            tot += batch_t["label"].size(0)
        tr_loss /= tot
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

        val_loss, c1, s1, h1, cn1, *_ = evaluate(model, dev_dl)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append((c1, s1, h1, cn1))

        print(
            f"Epoch {ep}: validation_loss = {val_loss:.4f} | CWA={c1:.3f} SWA={s1:.3f} HWA={h1:.3f} CNA={cn1:.3f}"
        )

    test_loss, c2, s2, h2, cn2, ys, yh, _ = evaluate(model, test_dl)
    print(
        f"TEST  : loss={test_loss:.4f} | CWA={c2:.3f} SWA={s2:.3f} HWA={h2:.3f} CNA={cn2:.3f}"
    )
    experiment_data["SPR_BENCH"]["predictions"] = yh
    experiment_data["SPR_BENCH"]["ground_truth"] = ys
    experiment_data["SPR_BENCH"]["metrics"]["test"] = (c2, s2, h2, cn2)


train(lr=2e-3, epochs=5)

# ------------------------------------------------ SAVE ---------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
