import os, math, time, random, pathlib, numpy as np, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- reproducibility ----------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ---------- working directory ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment dict ----------
experiment_data = {
    "no_glyph_clustering": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- helpers ----------
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
    splits = {"train": 500, "dev": 100, "test": 100}
    shapes, colors, rng = "ABCD", "1234", random.Random(0)

    def mkseq():
        return " ".join(
            rng.choice(shapes) + rng.choice(colors) for _ in range(rng.randint(4, 8))
        )

    for split, n in splits.items():
        with open(root / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                seq = mkseq()
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

    return DatasetDict({k: _ld(root / f"{k}.csv") for k in ["train", "dev", "test"]})


# ---------- locate dataset ----------
spr_root = _discover_spr_path()
if spr_root is None:
    spr_root = pathlib.Path(working_dir) / "SPR_BENCH_TOY"
    _create_toy_dataset(spr_root)
print("Using SPR_BENCH folder:", spr_root)
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})


# ---------- metrics ----------
def count_color_variety(seq):
    return len({tok[1:] for tok in seq.strip().split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.strip().split() if tok})


def color_weighted_accuracy(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == pr else 0 for wt, t, pr in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == pr else 0 for wt, t, pr in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


def cluster_normalised_accuracy(clists, y, p):
    from collections import defaultdict

    tot, corr = defaultdict(int), defaultdict(int)
    for cl, t, pr in zip(clists, y, p):
        if not cl:
            continue
        dom = Counter(cl).most_common(1)[0][0]
        tot[dom] += 1
        if t == pr:
            corr[dom] += 1
    if not tot:
        return 0.0
    return sum(corr[c] / tot[c] for c in tot) / len(tot)


# ---------- full-vocab mapping ----------
all_glyphs = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
glyph2id = {g: i + 1 for i, g in enumerate(all_glyphs)}  # 0=PAD
unk_id = len(glyph2id) + 1
vocab_size = len(glyph2id) + 2  # +PAD +UNK
print(f"Full vocabulary size (incl PAD/UNK): {vocab_size}")


# ---------- dataset ----------
class SPRFullVocab(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [int(x) for x in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [glyph2id.get(tok, unk_id) for tok in self.seqs[idx].split()]
        return {
            "input": torch.tensor(ids, dtype=torch.long),
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
    SPRFullVocab(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRFullVocab(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRFullVocab(spr["test"]), batch_size=512, shuffle=False, collate_fn=collate
)

num_labels = len(set(spr["train"]["label"]))


# ---------- model ----------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, nclass, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.fc = nn.Linear(d_model, nclass)

    def forward(self, x):
        src = self.emb(x).transpose(0, 1)  # [seq,batch,emb]
        L = src.size(0)
        src = src + self.pos[:L]
        pad_mask = x == 0
        enc = self.encoder(src, src_key_padding_mask=pad_mask)
        enc = enc.masked_fill(pad_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        summed = enc.sum(dim=0)
        lens = (~pad_mask).sum(dim=1).unsqueeze(-1).clamp(min=1)
        return self.fc(summed / lens)


criterion = nn.CrossEntropyLoss()


# ---------- evaluation ----------
def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, clists = [], [], [], []
    tot_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            t_batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(t_batch["input"])
            loss = criterion(logits, t_batch["label"])
            bs = t_batch["label"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(t_batch["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
            clists.extend(batch["cluster_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    cna = cluster_normalised_accuracy(clists, gts, preds)
    return tot_loss / n, cwa, swa, hwa, cna, preds, gts


# ---------- training ----------
def train(lr=2e-3, epochs=5):
    model = TransformerClassifier(vocab_size, 64, 4, 2, num_labels, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, seen = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            t_batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(t_batch["input"])
            loss = criterion(logits, t_batch["label"])
            loss.backward()
            opt.step()
            bs = t_batch["label"].size(0)
            ep_loss += loss.item() * bs
            seen += bs
        tr_loss = ep_loss / seen
        experiment_data["no_glyph_clustering"]["SPR_BENCH"]["losses"]["train"].append(
            (lr, ep, tr_loss)
        )
        val_loss, cwa, swa, hwa, cna, _, _ = evaluate(model, dev_loader)
        experiment_data["no_glyph_clustering"]["SPR_BENCH"]["losses"]["val"].append(
            (lr, ep, val_loss)
        )
        experiment_data["no_glyph_clustering"]["SPR_BENCH"]["metrics"]["val"].append(
            (lr, ep, cwa, swa, hwa, cna)
        )
        print(
            f"Epoch {ep}: val_loss={val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )
    # test
    test_loss, cwa, swa, hwa, cna, preds, gts = evaluate(model, test_loader)
    experiment_data["no_glyph_clustering"]["SPR_BENCH"]["predictions"] = preds
    experiment_data["no_glyph_clustering"]["SPR_BENCH"]["ground_truth"] = gts
    experiment_data["no_glyph_clustering"]["SPR_BENCH"]["metrics"]["test"] = (
        lr,
        cwa,
        swa,
        hwa,
        cna,
    )
    print(
        f"TEST | loss={test_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
    )


train(lr=2e-3, epochs=5)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
