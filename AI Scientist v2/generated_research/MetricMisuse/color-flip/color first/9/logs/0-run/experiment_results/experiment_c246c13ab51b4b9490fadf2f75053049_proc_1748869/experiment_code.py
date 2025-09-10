# ---------- imports ----------
import os, math, time, random, pathlib, numpy as np, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ---------- experiment dict ----------
experiment_data = {
    "multi_head": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
    "single_head": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- locate / build dataset ----------
def _discover_spr_path() -> pathlib.Path | None:
    env = os.getenv("SPR_DATA")
    if env and pathlib.Path(env).expanduser().exists():
        return pathlib.Path(env).expanduser()
    hard = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if hard.exists():
        return hard
    cur = pathlib.Path.cwd()
    for p in [cur, *cur.parents]:
        cand = p / "SPR_BENCH"
        if cand.exists():
            return cand
    return None


def _create_toy_dataset(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train": 500, "dev": 100, "test": 100}
    shapes, colors = "ABCD", "1234"
    rng = random.Random(0)

    def make():
        return " ".join(
            rng.choice(shapes) + rng.choice(colors) for _ in range(rng.randint(4, 8))
        )

    for split, n in splits.items():
        with open(root / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                seq = make()
                label = int(
                    sum(tok[0] == "A" for tok in seq.split()) > len(seq.split()) / 2
                )
                f.write(f"{i},{seq},{label}\n")
    print("Toy dataset created at", root)


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    ld = lambda p: load_dataset(
        "csv", data_files=str(p), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(
        {
            "train": ld(root / "train.csv"),
            "dev": ld(root / "dev.csv"),
            "test": ld(root / "test.csv"),
        }
    )


spr_root = _discover_spr_path()
if spr_root is None:
    spr_root = pathlib.Path("working/SPR_BENCH_TOY")
    _create_toy_dataset(spr_root)
print("Using SPR_BENCH folder:", spr_root)
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# ---------- metric helpers ----------
def count_color_variety(seq):
    return len({t[1:] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split()})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


def cluster_normalised_accuracy(cluster_seqs, y_true, y_pred):
    from collections import defaultdict

    tot = defaultdict(int)
    corr = defaultdict(int)
    for clist, t, p in zip(cluster_seqs, y_true, y_pred):
        if not clist:
            continue
        dom = Counter(clist).most_common(1)[0][0]
        tot[dom] += 1
        if t == p:
            corr[dom] += 1
    if not tot:
        return 0.0
    return sum(corr[c] / tot[c] for c in tot) / len(tot)


# ---------- glyph clustering ----------
all_glyphs = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
k_clusters = min(16, len(all_glyphs)) or 1
features = np.stack(
    [
        [ord(tok[0]), np.mean([ord(c) for c in tok[1:]]) if len(tok) > 1 else 0.0]
        for tok in all_glyphs
    ]
)
labels = KMeans(n_clusters=k_clusters, random_state=0, n_init="auto").fit_predict(
    features
)
glyph2cluster = {g: int(c) + 1 for g, c in zip(all_glyphs, labels)}  # 0 for PAD
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")


# ---------- dataset ----------
class SPRClustered(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [int(x) for x in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        clust_ids = [glyph2cluster.get(tok, 0) for tok in self.seqs[idx].split()]
        return {
            "input": torch.tensor(clust_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lens = [len(b["input"]) for b in batch]
    max_len = max(lens)
    pad = lambda t: torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.long)])
    return {
        "input": torch.stack([pad(b["input"]) for b in batch]),
        "len": torch.tensor(lens),
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
        lyr = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(lyr, nlayers)
        self.fc = nn.Linear(d_model, nclass)

    def forward(self, x):
        src = self.emb(x).transpose(0, 1)
        src_len = src.size(0)
        src = src + self.pos[:src_len]
        pad_mask = x == 0
        enc = self.encoder(src, src_key_padding_mask=pad_mask)
        enc = enc.masked_fill(pad_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        summed = enc.sum(0)
        lens = (~pad_mask).sum(1, keepdim=True).clamp(min=1)
        pooled = summed / lens
        return self.fc(pooled)


criterion = nn.CrossEntropyLoss()


# ---------- evaluation ----------
def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, clseq = [], [], [], []
    tot_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch_torch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch_torch["input"])
            loss = criterion(logits, batch_torch["label"])
            bs = batch_torch["label"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch_torch["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
            clseq.extend(batch["cluster_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    cna = cluster_normalised_accuracy(clseq, gts, preds)
    return tot_loss / n, cwa, swa, hwa, cna, preds, gts


# ---------- training wrapper ----------
def run_experiment(tag, nhead, lr=2e-3, epochs=5):
    store = experiment_data[tag]["SPR_BENCH"]
    model = TransformerClassifier(
        vocab_size, d_model=64, nhead=nhead, nlayers=2, nclass=num_labels, max_len=64
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, seen = 0.0, 0
        for batch in train_loader:
            optim.zero_grad()
            batch_torch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch_torch["input"])
            loss = criterion(logits, batch_torch["label"])
            loss.backward()
            optim.step()
            bs = batch_torch["label"].size(0)
            ep_loss += loss.item() * bs
            seen += bs
        tr_loss = ep_loss / seen
        store["losses"]["train"].append((lr, ep, tr_loss))
        # validation
        val_loss, cwa, swa, hwa, cna, _, _ = evaluate(model, dev_loader)
        store["losses"]["val"].append((lr, ep, val_loss))
        store["metrics"]["val"].append((lr, ep, cwa, swa, hwa, cna))
        print(
            f"[{tag}] Epoch {ep}: val_loss={val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )
    # test
    test_loss, cwa, swa, hwa, cna, preds, gts = evaluate(model, test_loader)
    print(
        f"[{tag}] TEST: loss={test_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
    )
    store["predictions"] = preds
    store["ground_truth"] = gts
    store["metrics"]["test"] = (lr, cwa, swa, hwa, cna)


# ---------- run both configurations ----------
run_experiment("multi_head", nhead=4, lr=2e-3, epochs=5)
run_experiment("single_head", nhead=1, lr=2e-3, epochs=5)

# ---------- save ----------
out_file = "experiment_data.npy"
np.save(out_file, experiment_data)
print("Saved experiment data to", out_file)
