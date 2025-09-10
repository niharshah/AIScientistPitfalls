import os, math, time, random, pathlib, numpy as np, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ----------------------- experiment bookkeeping -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "no_positional_encoding": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ----------------------- device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------- data helpers -----------------------
def _discover_spr_path() -> pathlib.Path | None:
    env = os.getenv("SPR_DATA")
    if env and pathlib.Path(env).expanduser().exists():
        return pathlib.Path(env).expanduser()
    fixed = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if fixed.exists():
        return fixed
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        c = p / "SPR_BENCH"
        if c.exists():
            return c
    return None


def _create_toy_dataset(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train": 500, "dev": 100, "test": 100}
    rng = random.Random(0)
    shapes, colors = "ABCD", "1234"

    def make_seq():
        l = rng.randint(4, 8)
        return " ".join(rng.choice(shapes) + rng.choice(colors) for _ in range(l))

    for split, n in splits.items():
        with open(root / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                seq = make_seq()
                label = int(
                    sum(tok[0] == "A" for tok in seq.split()) > len(seq.split()) / 2
                )
                f.write(f"{i},{seq},{label}\n")
    print("Toy SPR created at", root)


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_path):  # single CSV to HF dataset
        return load_dataset(
            "csv", data_files=str(csv_path), split="train", cache_dir=".cache_dsets"
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

spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# ----------------------- metrics -----------------------
def count_color_variety(seq: str):
    return len({t[1:] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq: str):
    return len({t[0] for t in seq.split() if t})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / sum(w)
        if sum(w)
        else 0.0
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / sum(w)
        if sum(w)
        else 0.0
    )


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if cwa + swa else 0.0


def cluster_normalised_accuracy(cluster_seqs, y_true, y_pred):
    from collections import defaultdict

    total, correct = defaultdict(int), defaultdict(int)
    for clist, t, p in zip(cluster_seqs, y_true, y_pred):
        if not clist:
            continue
        dom = Counter(clist).most_common(1)[0][0]
        total[dom] += 1
        if t == p:
            correct[dom] += 1
    if not total:
        return 0.0
    return sum(correct[c] / total[c] for c in total) / len(total)


# ----------------------- glyph clustering -----------------------
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
glyph2cluster = {g: int(c) + 1 for g, c in zip(all_glyphs, labels)}  # 0 = PAD
print(f"Clustered {len(all_glyphs)} glyphs into {k_clusters} clusters.")


# ----------------------- dataset -----------------------
class SPRClustered(Dataset):
    def __init__(self, ds):
        self.seqs, self.labels = ds["sequence"], [int(l) for l in ds["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [glyph2cluster.get(tok, 0) for tok in self.seqs[idx].split()]
        return {
            "input": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lens = [len(b["input"]) for b in batch]
    maxlen = max(lens)
    padded = [
        torch.cat([b["input"], torch.zeros(maxlen - len(b["input"]), dtype=torch.long)])
        for b in batch
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

num_labels, vocab_size = len(set(spr["train"]["label"])), k_clusters + 1


# ----------------------- model -----------------------
class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab, d_model, nhead, nlayers, nclass, max_len=64, use_pos=True
    ):
        super().__init__()
        self.use_pos = use_pos
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(
            torch.randn(max_len, 1, d_model)
        )  # kept for shape even if not used
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 2 * d_model, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.fc = nn.Linear(d_model, nclass)

    def forward(self, x):
        src = self.emb(x).transpose(0, 1)  # [seq,batch,emb]
        if self.use_pos:
            src = src + self.pos[: src.size(0)]
        pad_mask = x == 0
        enc = self.encoder(src, src_key_padding_mask=pad_mask)
        enc = enc.masked_fill(pad_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        summed = enc.sum(0)
        lens = (~pad_mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(summed / lens)


criterion = nn.CrossEntropyLoss()


# ----------------------- evaluation -----------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, cseqs = [], [], [], []
    tot_loss, seen = 0.0, 0
    for batch in loader:
        bt = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = model(bt["input"])
        loss = criterion(logits, bt["label"])
        bs = bt["label"].size(0)
        tot_loss += loss.item() * bs
        seen += bs
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(bt["label"].cpu().tolist())
        seqs.extend(batch["raw_seq"])
        cseqs.extend(batch["cluster_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    cna = cluster_normalised_accuracy(cseqs, gts, preds)
    return tot_loss / seen, cwa, swa, hwa, cna, preds, gts


# ----------------------- training (ablation) -----------------------
def train_ablation(lr=2e-3, epochs=5):
    model = TransformerClassifier(
        vocab_size, 64, 4, 2, num_labels, max_len=64, use_pos=False
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, seen = 0.0, 0
        for batch in train_loader:
            optim.zero_grad()
            bt = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            loss = criterion(model(bt["input"]), bt["label"])
            loss.backward()
            optim.step()
            bs = bt["label"].size(0)
            ep_loss += loss.item() * bs
            seen += bs
        tr_loss = ep_loss / seen
        experiment_data["no_positional_encoding"]["SPR_BENCH"]["losses"][
            "train"
        ].append((lr, ep, tr_loss))

        val_loss, cwa, swa, hwa, cna, _, _ = evaluate(model, dev_loader)
        experiment_data["no_positional_encoding"]["SPR_BENCH"]["losses"]["val"].append(
            (lr, ep, val_loss)
        )
        experiment_data["no_positional_encoding"]["SPR_BENCH"]["metrics"]["val"].append(
            (lr, ep, cwa, swa, hwa, cna)
        )
        print(
            f"[EP {ep}] val_loss={val_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )

    # final test
    test_loss, cwa, swa, hwa, cna, preds, gts = evaluate(model, test_loader)
    experiment_data["no_positional_encoding"]["SPR_BENCH"]["predictions"] = preds
    experiment_data["no_positional_encoding"]["SPR_BENCH"]["ground_truth"] = gts
    experiment_data["no_positional_encoding"]["SPR_BENCH"]["metrics"]["test"] = (
        lr,
        cwa,
        swa,
        hwa,
        cna,
    )
    print(
        f"TEST | loss={test_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
    )


train_ablation(lr=2e-3, epochs=5)

# ----------------------- save -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
