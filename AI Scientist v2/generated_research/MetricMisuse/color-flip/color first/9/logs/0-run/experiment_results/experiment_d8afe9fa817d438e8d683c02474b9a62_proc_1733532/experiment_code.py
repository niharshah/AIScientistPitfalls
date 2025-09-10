import os, pathlib, random, math, time
import numpy as np, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------
# WORK DIR + DEVICE
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# EXPERIMENT DATA CONTAINER
# ------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------
# DATASET LOCATION HELPER
# ------------------------------------------------------------
REQUIRED_CSV = {"train.csv", "dev.csv", "test.csv"}


def find_spr_bench_dir() -> pathlib.Path:
    """Return a pathlib.Path pointing to the SPR_BENCH folder."""
    # 1) explicit environment variable
    env_path = os.getenv("SPR_DATA")
    if env_path:
        p = pathlib.Path(env_path).expanduser().resolve()
        if REQUIRED_CSV.issubset({f.name for f in p.iterdir()}):
            print(f"Found SPR_BENCH via $SPR_DATA at {p}")
            return p

    # 2) walk up parents from CWD
    cwd = pathlib.Path(os.getcwd()).resolve()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "SPR_BENCH"
        if candidate.exists() and REQUIRED_CSV.issubset(
            {f.name for f in candidate.iterdir()}
        ):
            print(f"Found SPR_BENCH at {candidate}")
            return candidate

    # 3) fallback to absolute path used in original spec
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH").expanduser()
    if fallback.exists() and REQUIRED_CSV.issubset(
        {f.name for f in fallback.iterdir()}
    ):
        print(f"Found SPR_BENCH at fallback location {fallback}")
        return fallback

    raise FileNotFoundError(
        "Unable to locate SPR_BENCH directory. "
        "Set $SPR_DATA, or place SPR_BENCH next to the current working directory."
    )


# ------------------------------------------------------------
# HF LOADER (unchanged)
# ------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


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
    cluster_total, cluster_correct = Counter(), Counter()
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


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
DATA_PATH = find_spr_bench_dir()
spr = load_spr_bench(DATA_PATH)
print("Loaded dataset sizes:", {k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------
# GLYPH CLUSTERING
# ------------------------------------------------------------
all_glyphs = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
k_clusters = 16
features = np.stack(
    [
        [ord(tok[0]), np.mean([ord(c) for c in tok[1:]]) if len(tok) > 1 else 0.0]
        for tok in all_glyphs
    ]
)
labels = KMeans(n_clusters=k_clusters, random_state=0, n_init=10).fit_predict(features)
glyph2cluster = {
    g: int(c) + 1 for g, c in zip(all_glyphs, labels)
}  # 0 reserved for PAD
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")


# ------------------------------------------------------------
# TORCH DATASET
# ------------------------------------------------------------
class SPRClustered(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [int(x) for x in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        cluster_ids = [glyph2cluster.get(tok, 0) for tok in self.seqs[idx].split()]
        return {
            "input": torch.tensor(cluster_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
            "cluster_ids": cluster_ids,  # keep python list for CNA
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
        "cluster_seq": [b["cluster_ids"] for b in batch],
    }


batch_sz = 256
train_loader = DataLoader(
    SPRClustered(spr["train"]), batch_size=batch_sz, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRClustered(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRClustered(spr["test"]), batch_size=512, shuffle=False, collate_fn=collate
)

num_labels = len(set(spr["train"]["label"]))
vocab_size = k_clusters + 1  # + PAD


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, nclass, max_len=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(d_model, nclass)

    def forward(self, x):
        # x: [batch, seq]
        src = self.emb(x).transpose(0, 1)  # [seq,batch,emb]
        seq_len = src.size(0)
        src = src + self.pos[:seq_len]
        pad_mask = x == 0  # [batch,seq]
        out = self.encoder(src, src_key_padding_mask=pad_mask)
        out = out.masked_fill(pad_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        summed = out.sum(dim=0)
        lens = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = summed / lens
        return self.fc(mean_pool)


criterion = nn.CrossEntropyLoss()


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, clusters = [], [], [], []
    tot_loss, seen = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input"])
            loss = criterion(logits, batch["label"])
            bs = batch["label"].size(0)
            tot_loss += loss.item() * bs
            seen += bs
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
            clusters.extend(batch["cluster_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    cna = cluster_normalised_accuracy(clusters, gts, preds)
    return tot_loss / seen, cwa, swa, hwa, cna, preds, gts


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def train(lr=2e-3, epochs=5):
    model = TransformerClassifier(
        vocab_size, d_model=64, nhead=4, nlayers=2, nclass=num_labels
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            seen += bs
        tr_loss = total_loss / seen
        experiment_data["SPR_BENCH"]["losses"]["train"].append((lr, ep, tr_loss))

        val_loss, cwa, swa, hwa, cna, _, _ = evaluate(model, dev_loader)
        experiment_data["SPR_BENCH"]["losses"]["val"].append((lr, ep, val_loss))
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            (lr, ep, cwa, swa, hwa, cna)
        )
        print(
            f"Epoch {ep}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )

    test_loss, cwa, swa, hwa, cna, preds, gts = evaluate(model, test_loader)
    experiment_data["SPR_BENCH"]["metrics"]["test"] = (lr, cwa, swa, hwa, cna)
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts
    print(
        f"TEST | loss={test_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
    )

    # save
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
train(lr=2e-3, epochs=5)
