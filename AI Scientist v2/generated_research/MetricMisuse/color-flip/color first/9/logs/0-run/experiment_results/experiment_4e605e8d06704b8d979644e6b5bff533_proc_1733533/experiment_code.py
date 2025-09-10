import os, math, time, random, pathlib, numpy as np, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ---------- working directory ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- dataset helpers ----------
def _discover_spr_path() -> pathlib.Path | None:
    """
    Try multiple heuristics to locate the SPR_BENCH folder.
    Returns a pathlib.Path or None if nothing is found.
    """
    # 1. explicit environment variable
    env_path = os.getenv("SPR_DATA")
    if env_path and pathlib.Path(env_path).expanduser().exists():
        return pathlib.Path(env_path).expanduser()

    # 2. absolute path seen in previous log
    hard_coded = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if hard_coded.exists():
        return hard_coded

    # 3. look for SPR_BENCH folder in current or parent dirs
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        candidate = parent / "SPR_BENCH"
        if candidate.exists():
            return candidate
    return None


def _create_toy_dataset(root: pathlib.Path):
    """
    Create a very small synthetic SPR-like dataset so that the
    rest of the pipeline can still run if real data is missing.
    """
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train": 500, "dev": 100, "test": 100}
    shapes = "ABCD"
    colors = "1234"
    rng = random.Random(0)

    def make_seq():
        length = rng.randint(4, 8)
        return " ".join(rng.choice(shapes) + rng.choice(colors) for _ in range(length))

    for split, nrows in splits.items():
        with open(root / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(nrows):
                seq = make_seq()
                # arbitrary rule: label 1 if majority shape is 'A', else 0
                label = int(
                    sum(tok[0] == "A" for tok in seq.split()) > len(seq.split()) / 2
                )
                f.write(f"{i},{seq},{label}\n")
    print(f"Created toy dataset in {root}")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    """
    Wrapper around HF load_dataset that produces a DatasetDict with
    'train'/'dev'/'test' splits even when given local single CSV files.
    """

    def _load(path_csv: pathlib.Path):
        return load_dataset(
            "csv",
            data_files=str(path_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load(root / "train.csv"),
            "dev": _load(root / "dev.csv"),
            "test": _load(root / "test.csv"),
        }
    )


# ---------- locate dataset ----------
spr_root = _discover_spr_path()
if spr_root is None:
    # No dataset found → build a tiny synthetic one inside working_dir
    spr_root = pathlib.Path(working_dir) / "SPR_BENCH_TOY"
    _create_toy_dataset(spr_root)

print("Using SPR_BENCH folder:", spr_root)
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# ---------- metric helpers ----------
def count_color_variety(sequence: str) -> int:
    return len({tok[1:] for tok in sequence.strip().split() if len(tok) > 1})


def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in sequence.strip().split() if tok})


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


def cluster_normalised_accuracy(seqs_clusters, y_true, y_pred):
    from collections import defaultdict

    cluster_total, cluster_correct = defaultdict(int), defaultdict(int)
    for clist, t, p in zip(seqs_clusters, y_true, y_pred):
        if not clist:
            continue
        dom = Counter(clist).most_common(1)[0][0]
        cluster_total[dom] += 1
        if t == p:
            cluster_correct[dom] += 1
    if not cluster_total:
        return 0.0
    per_cluster = [cluster_correct[c] / cluster_total[c] for c in cluster_total]
    return sum(per_cluster) / len(per_cluster)


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
glyph2cluster = {
    g: int(c) + 1 for g, c in zip(all_glyphs, labels)
}  # 0 reserved for PAD
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")


# ---------- dataset / dataloader ----------
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
    lengths = [len(b["input"]) for b in batch]
    max_len = max(lengths)
    padded = [
        torch.cat(
            [b["input"], torch.zeros(max_len - len(b["input"]), dtype=torch.long)]
        )
        for b in batch
    ]
    return {
        "input": torch.stack(padded),
        "len": torch.tensor(lengths, dtype=torch.long),
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
vocab_size = k_clusters + 1  # +1 for padding idx=0


# ---------- model ----------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, nclass, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.fc = nn.Linear(d_model, nclass)

    def forward(self, x):
        # x : [batch, seq]
        src = self.emb(x).transpose(0, 1)  # [seq,batch,emb]
        seq_len = src.size(0)
        src = src + self.pos[:seq_len]
        pad_mask = x == 0
        enc = self.encoder(src, src_key_padding_mask=pad_mask)
        enc = enc.masked_fill(pad_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        summed = enc.sum(dim=0)  # [batch,emb]
        lens = (~pad_mask).sum(dim=1).unsqueeze(-1).clamp(min=1)
        pooled = summed / lens
        return self.fc(pooled)


# ---------- evaluation ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, cluster_seqs = [], [], [], []
    tot_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            # Move tensors to device
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
            cluster_seqs.extend(batch["cluster_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    cna = cluster_normalised_accuracy(cluster_seqs, gts, preds)
    return tot_loss / n, cwa, swa, hwa, cna, preds, gts


# ---------- training ----------
def train(lr=2e-3, epochs=5):
    model = TransformerClassifier(
        vocab_size, d_model=64, nhead=4, nlayers=2, nclass=num_labels, max_len=64
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_torch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch_torch["input"])
            loss = criterion(logits, batch_torch["label"])
            loss.backward()
            optimizer.step()
            bs = batch_torch["label"].size(0)
            ep_loss += loss.item() * bs
            seen += bs
        tr_loss = ep_loss / seen
        experiment_data["SPR_BENCH"]["losses"]["train"].append((lr, ep, tr_loss))

        # validation
        val_loss, cwa, swa, hwa, cna, _, _ = evaluate(model, dev_loader)
        experiment_data["SPR_BENCH"]["losses"]["val"].append((lr, ep, val_loss))
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            (lr, ep, cwa, swa, hwa, cna)
        )
        print(
            f"Epoch {ep}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )

    # test evaluation
    test_loss, cwa, swa, hwa, cna, preds, gts = evaluate(model, test_loader)
    print(
        f"TEST | loss={test_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
    )
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts
    experiment_data["SPR_BENCH"]["metrics"]["test"] = (lr, cwa, swa, hwa, cna)


train(lr=2e-3, epochs=5)

# ---------- save results ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
