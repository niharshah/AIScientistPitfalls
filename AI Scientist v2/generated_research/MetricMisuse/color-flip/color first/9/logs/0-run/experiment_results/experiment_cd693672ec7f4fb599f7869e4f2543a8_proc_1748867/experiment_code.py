import os, math, time, random, pathlib, numpy as np, torch, torch.nn as nn
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ---------------------- experiment bookkeeping ----------------------
experiment_data = {
    "CLS_token_pooling": {  # ablation type
        # place-holders for two model variants, filled later
        "SPR_BENCH_MEAN": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "SPR_BENCH_CLS": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# ---------------------- misc helpers ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def _discover_spr_path() -> pathlib.Path | None:
    env_path = os.getenv("SPR_DATA")
    if env_path and pathlib.Path(env_path).expanduser().exists():
        return pathlib.Path(env_path).expanduser()
    hard_coded = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if hard_coded.exists():
        return hard_coded
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        candidate = parent / "SPR_BENCH"
        if candidate.exists():
            return candidate
    return None


def _create_toy_dataset(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train": 500, "dev": 100, "test": 100}
    shapes, colors, rng = "ABCD", "1234", random.Random(0)

    def make_seq():
        return " ".join(
            rng.choice(shapes) + rng.choice(colors) for _ in range(rng.randint(4, 8))
        )

    for split, nrows in splits.items():
        with open(root / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(nrows):
                seq = make_seq()
                label = int(
                    sum(tok[0] == "A" for tok in seq.split()) > len(seq.split()) / 2
                )
                f.write(f"{i},{seq},{label}\n")
    print("Created toy dataset at", root)


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(p: pathlib.Path):
        return load_dataset(
            "csv", data_files=str(p), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load(root / "train.csv"),
            "dev": _load(root / "dev.csv"),
            "test": _load(root / "test.csv"),
        }
    )


spr_root = _discover_spr_path()
if spr_root is None:
    spr_root = pathlib.Path(working_dir) / "SPR_BENCH_TOY"
    _create_toy_dataset(spr_root)
print("Using SPR_BENCH folder:", spr_root)
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# ---------------------- metrics ----------------------
def count_color_variety(seq: str):
    return len({tok[1:] for tok in seq.split()})


def count_shape_variety(seq: str):
    return len({tok[0] for tok in seq.split()})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if cwa + swa else 0


def cluster_normalised_accuracy(cluster_seqs, y_true, y_pred):
    total, correct = defaultdict(int), defaultdict(int)
    for clst, t, p in zip(cluster_seqs, y_true, y_pred):
        if not clst:
            continue
        dom = Counter(clst).most_common(1)[0][0]
        total[dom] += 1
        if t == p:
            correct[dom] += 1
    if not total:
        return 0.0
    return sum(correct[c] / total[c] for c in total) / len(total)


# ---------------------- glyph clustering ----------------------
all_glyphs = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
k_clusters = min(16, len(all_glyphs)) or 1
features = np.stack([[ord(t[0]), ord(t[1]) if len(t) > 1 else 0] for t in all_glyphs])
labels = KMeans(n_clusters=k_clusters, random_state=0, n_init="auto").fit_predict(
    features
)
glyph2cluster = {g: int(c) + 1 for g, c in zip(all_glyphs, labels)}  # 0 for PAD
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")


# ---------------------- dataset ----------------------
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
    padded = [
        torch.cat([b["input"], torch.zeros(max_len - l, dtype=torch.long)])
        for b, l in zip(batch, lens)
    ]
    return {
        "input": torch.stack(padded),
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
vocab_size = k_clusters + 1  # padding=0


# ---------------------- model ----------------------
class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab, d_model, nhead, nlayers, nclass, max_len=64, pooling="mean"
    ):
        super().__init__()
        self.pooling = pooling
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(
            torch.randn(max_len + (1 if pooling == "cls" else 0), 1, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, d_model)) if pooling == "cls" else None
        )
        self.fc = nn.Linear(d_model, nclass)

    def forward(self, x):
        # x: [batch, seq_len]
        src = self.emb(x).transpose(0, 1)  # [seq,batch,dim]
        if self.pooling == "cls":
            cls_tok = self.cls_token.repeat(1, src.size(1), 1)  # [1,batch,dim]
            src = torch.cat([cls_tok, src], dim=0)  # prepend
        src = src + self.pos[: src.size(0)]
        pad_mask = x == 0
        if self.pooling == "cls":
            pad_mask = torch.cat(
                [
                    torch.zeros(pad_mask.size(0), 1, dtype=torch.bool, device=x.device),
                    pad_mask,
                ],
                dim=1,
            )
        enc = self.encoder(src, src_key_padding_mask=pad_mask)
        if self.pooling == "mean":
            enc = enc.masked_fill(pad_mask.transpose(0, 1).unsqueeze(-1), 0.0)
            summed = enc.sum(dim=0)
            lens = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = summed / lens
        else:  # CLS pooling
            pooled = enc[0]  # [batch,dim]
        return self.fc(pooled)


# ---------------------- train / evaluate ----------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, cluster_seqs = [], [], [], []
    tot_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["input"])
            loss = criterion(logits, batch_t["label"])
            bs = batch_t["label"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch_t["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
            cluster_seqs.extend(batch["cluster_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    cna = cluster_normalised_accuracy(cluster_seqs, gts, preds)
    return tot_loss / n, cwa, swa, hwa, cna, preds, gts


def train_variant(pooling_mode, exp_key, lr=2e-3, epochs=5):
    print("\n=== Training variant:", pooling_mode, "===\n")
    model = TransformerClassifier(
        vocab_size, 64, 4, 2, num_labels, max_len=64, pooling=pooling_mode
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0
        seen = 0
        for batch in train_loader:
            optim.zero_grad()
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["input"])
            loss = criterion(logits, batch_t["label"])
            loss.backward()
            optim.step()
            bs = batch_t["label"].size(0)
            ep_loss += loss.item() * bs
            seen += bs
        tr_loss = ep_loss / seen
        experiment_data["CLS_token_pooling"][exp_key]["losses"]["train"].append(
            (lr, ep, tr_loss)
        )

        val_loss, cwa, swa, hwa, cna, _, _ = evaluate(model, dev_loader)
        experiment_data["CLS_token_pooling"][exp_key]["losses"]["val"].append(
            (lr, ep, val_loss)
        )
        experiment_data["CLS_token_pooling"][exp_key]["metrics"]["val"].append(
            (lr, ep, cwa, swa, hwa, cna)
        )
        print(
            f"[{pooling_mode}] Epoch {ep} | val_loss={val_loss:.4f} "
            f"CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )
    # test
    test_loss, cwa, swa, hwa, cna, preds, gts = evaluate(model, test_loader)
    print(
        f"[{pooling_mode}] TEST | loss={test_loss:.4f} CWA={cwa:.3f} "
        f"SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
    )
    ed = experiment_data["CLS_token_pooling"][exp_key]
    ed["predictions"] = preds
    ed["ground_truth"] = gts
    ed["metrics"]["test"] = (lr, cwa, swa, hwa, cna)


# ---------------------- run both variants ----------------------
train_variant("mean", "SPR_BENCH_MEAN", lr=2e-3, epochs=5)
train_variant("cls", "SPR_BENCH_CLS", lr=2e-3, epochs=5)

# ---------------------- save ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
