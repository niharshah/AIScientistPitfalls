# Random Glyph Clustering Ablation – self-contained script
import os, math, time, random, pathlib, numpy as np, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- experiment bookkeeping ----------
experiment_data = {
    "RandomCluster": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}
ABLATION = "RandomCluster"
DSNAME = "SPR_BENCH"

# ---------- misc ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- dataset helpers ----------
def _discover_spr_path() -> pathlib.Path | None:
    env_path = os.getenv("SPR_DATA")
    if env_path and pathlib.Path(env_path).expanduser().exists():
        return pathlib.Path(env_path).expanduser()
    hard_coded = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if hard_coded.exists():
        return hard_coded
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        cand = parent / "SPR_BENCH"
        if cand.exists():
            return cand
    return None


def _create_toy_dataset(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    splits, shapes, colors = {"train": 500, "dev": 100, "test": 100}, "ABCD", "1234"
    rng = random.Random(0)
    for split, nrows in splits.items():
        with open(root / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(nrows):
                seq = " ".join(
                    rng.choice(shapes) + rng.choice(colors)
                    for _ in range(rng.randint(4, 8))
                )
                label = int(
                    sum(tok[0] == "A" for tok in seq.split()) > len(seq.split()) / 2
                )
                f.write(f"{i},{seq},{label}\n")
    print("Created toy dataset at", root)


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(p):
        return load_dataset(
            "csv", data_files=str(p), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _ld(root / "train.csv"),
            "dev": _ld(root / "dev.csv"),
            "test": _ld(root / "test.csv"),
        }
    )


spr_root = _discover_spr_path()
if spr_root is None:
    spr_root = pathlib.Path(working_dir) / "SPR_BENCH_TOY"
    _create_toy_dataset(spr_root)
print("Using SPR_BENCH folder:", spr_root)
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# ---------- metric helpers ----------
def count_color_variety(seq):
    return len({t[1:] for t in seq.strip().split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.strip().split() if t})


def color_weighted_accuracy(seqs, y, g):
    w = [count_color_variety(s) for s in seqs]
    return (
        sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / sum(w) if sum(w) else 0.0
    )


def shape_weighted_accuracy(seqs, y, g):
    w = [count_shape_variety(s) for s in seqs]
    return (
        sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / sum(w) if sum(w) else 0.0
    )


def harmonic_weighted_accuracy(c, s):
    return 2 * c * s / (c + s) if c + s else 0.0


def cluster_normalised_accuracy(clists, y, g):
    from collections import defaultdict

    tot = defaultdict(int)
    corr = defaultdict(int)
    for cl, yt, yp in zip(clists, y, g):
        if not cl:
            continue
        dom = Counter(cl).most_common(1)[0][0]
        tot[dom] += 1
        if yt == yp:
            corr[dom] += 1
    return np.mean([corr[c] / tot[c] for c in tot]) if tot else 0.0


# ---------- random glyph clustering ----------
all_glyphs = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
k_clusters = min(16, len(all_glyphs)) or 1
rng = random.Random(0)
glyphs_shuffled = all_glyphs[:]
rng.shuffle(glyphs_shuffled)
glyph2cluster = {
    g: (i % k_clusters) + 1 for i, g in enumerate(glyphs_shuffled)
}  # 0 reserved for PAD
print(f"Randomly assigned {len(all_glyphs)} glyphs into {k_clusters} clusters.")


# ---------- dataset ----------
class SPRClustered(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [int(x) for x in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        clust = [glyph2cluster.get(tok, 0) for tok in self.seqs[idx].split()]
        return {
            "input": torch.tensor(clust, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lens = [len(b["input"]) for b in batch]
    max_len = max(lens)
    padded = [
        torch.cat(
            [b["input"], torch.zeros(max_len - len(b["input"]), dtype=torch.long)]
        )
        for b in batch
    ]
    return {
        "input": torch.stack(padded),
        "len": torch.tensor(lens, dtype=torch.long),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
        "cluster_seq": [b["input"].tolist() for b in batch],
    }


batch_size = 256 if len(spr["train"]) > 256 else 64
train_loader = DataLoader(
    SPRClustered(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRClustered(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRClustered(spr["test"]), batch_size=512, shuffle=False, collate_fn=collate
)

num_labels = len(set(spr["train"]["label"]))
vocab_size = k_clusters + 1


# ---------- model ----------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, nclass, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 2 * d_model, batch_first=False
        )
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.fc = nn.Linear(d_model, nclass)

    def forward(self, x):
        src = self.emb(x).transpose(0, 1)
        src = src + self.pos[: src.size(0)]
        mask = x == 0
        h = self.enc(src, src_key_padding_mask=mask)
        h = h.masked_fill(mask.transpose(0, 1).unsqueeze(-1), 0.0)
        pooled = h.sum(0) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(pooled)


criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds = []
    gts = []
    seqs = []
    clists = []
    tot_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(bt["input"])
            loss = criterion(logits, bt["label"])
            bs = bt["label"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(bt["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
            clists.extend(batch["cluster_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    cna = cluster_normalised_accuracy(clists, gts, preds)
    return tot_loss / n, cwa, swa, hwa, cna, preds, gts


def train(lr=2e-3, epochs=5):
    model = TransformerClassifier(vocab_size, 64, 4, 2, num_labels, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0
        seen = 0
        for batch in train_loader:
            opt.zero_grad()
            bt = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            loss = criterion(model(bt["input"]), bt["label"])
            bs = bt["label"].size(0)
            loss.backward()
            opt.step()
            tot += loss.item() * bs
            seen += bs
        tr_loss = tot / seen
        experiment_data[ABLATION][DSNAME]["losses"]["train"].append((lr, ep, tr_loss))
        val_loss, cwa, swa, hwa, cna, _, _ = evaluate(model, dev_loader)
        experiment_data[ABLATION][DSNAME]["losses"]["val"].append((lr, ep, val_loss))
        experiment_data[ABLATION][DSNAME]["metrics"]["val"].append(
            (lr, ep, cwa, swa, hwa, cna)
        )
        print(
            f"Epoch {ep}: val_loss={val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )
    test_loss, cwa, swa, hwa, cna, preds, gts = evaluate(model, test_loader)
    print(
        f"TEST | loss={test_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
    )
    experiment_data[ABLATION][DSNAME]["predictions"] = preds
    experiment_data[ABLATION][DSNAME]["ground_truth"] = gts
    experiment_data[ABLATION][DSNAME]["metrics"]["test"] = (lr, cwa, swa, hwa, cna)


train(lr=2e-3, epochs=5)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
