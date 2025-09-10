import os, pathlib, random, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple

# ------------------ house-keeping & GPU --------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ------------------ locate SPR_BENCH ------------------------ #
def find_spr_bench_root() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_ROOT")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    here = pathlib.Path.cwd()
    for p in [here] + list(here.parents):
        if (p / "SPR_BENCH/train.csv").exists():
            return p / "SPR_BENCH"
    for p in [
        pathlib.Path.home() / d for d in ["SPR_BENCH", "AI-Scientist-v2/SPR_BENCH"]
    ]:
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError("SPR_BENCH not found")


DATA_PATH = find_spr_bench_root()


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split):
        return load_dataset(
            "csv",
            data_files=str(root / f"{split}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(s) for s in ["train", "dev", "test"]})


spr = load_spr_bench(DATA_PATH)
num_classes = len(set(spr["train"]["label"]))


# ----------------- helper evaluation metrics ---------------- #
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split()))


def color_weighted_accuracy(seqs, y, g):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y, g)]
    return sum(c) / sum(w) if sum(w) else 0


def shape_weighted_accuracy(seqs, y, g):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y, g)]
    return sum(c) / sum(w) if sum(w) else 0


def harmonic_csa(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ----------------- 1. build glyph vocab --------------------- #
vocab = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
g2i = {g: i for i, g in enumerate(vocab)}
V = len(vocab)
print(f"Vocabulary size: {V}")

# ----------------- 2. co-occurrence + SVD embeddings -------- #
window = 2
co_mat = np.zeros((V, V), dtype=np.float32)
for seq in spr["train"]["sequence"]:
    toks = [g2i[t] for t in seq.split()]
    for idx, t in enumerate(toks):
        for j in range(max(0, idx - window), min(len(toks), idx + window + 1)):
            if j != idx:
                co_mat[t, toks[j]] += 1.0
tot_pairs = co_mat.sum()
row_sum = co_mat.sum(axis=1, keepdims=True)
col_sum = co_mat.sum(axis=0, keepdims=True)
pmi = np.log((co_mat * tot_pairs + 1e-8) / (row_sum * col_sum + 1e-8))
ppmi = np.maximum(pmi, 0)
u, s, vt = np.linalg.svd(ppmi, full_matrices=False)
emb_dim = 32
glyph_emb = u[:, :emb_dim] * np.sqrt(s[:emb_dim])

# ----------------- 3. K-means clustering -------------------- #
k_clusters = 16
kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=20)
clusters = kmeans.fit_predict(glyph_emb)
glyph_to_cluster = {g: c for g, c in zip(vocab, clusters)}
print(f"Clustered {V} glyphs into {k_clusters} clusters")

# ---------- train-set glyph-cluster pairs for SNWA ---------- #
train_pairs = {(tok, glyph_to_cluster[tok]) for tok in vocab}  # all appear in train


def sequence_novelty_weight(seq: str) -> float:
    total, nov = 0, 0
    for tok in seq.split():
        total += 1
        if (tok, glyph_to_cluster[tok]) not in train_pairs:
            nov += 1
    return 1.0 + (nov / total if total else 0)


def snwa(seqs, y, g):
    w = [sequence_novelty_weight(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y, g)]
    return sum(c) / sum(w) if sum(w) else 0


# ----------------- 4. dataset & dataloader ------------------ #
PAD_IDX = k_clusters


def seq_to_cluster_ids(seq):
    return [glyph_to_cluster[t] for t in seq.split()]


class SPRSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.ids = [seq_to_cluster_ids(s) for s in self.seqs]
        self.labels = torch.tensor(split["label"], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"seq": self.ids[idx], "label": self.labels[idx]}


def collate(batch):
    lens = [len(b["seq"]) for b in batch]
    max_len = max(lens)
    seq_tensor = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for i, b in enumerate(batch):
        seq_tensor[i, : len(b["seq"])] = torch.tensor(b["seq"], dtype=torch.long)
    return {
        "seq": seq_tensor,
        "len": torch.tensor(lens, dtype=torch.long),
        "label": torch.stack([b["label"] for b in batch]),
    }


batch_size = 128
train_loader = DataLoader(
    SPRSet(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRSet(spr["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRSet(spr["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ----------------- 5. Transformer classifier --------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x):  # x: (B,L,D)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self, n_clusters, pad_idx, num_classes, d_model=64, nhead=4, nlayers=2
    ):
        super().__init__()
        self.emb = nn.Embedding(n_clusters + 1, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        x = self.emb(x)
        x = self.pos(x)
        x = self.enc(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)  # zero out pads
        pooled = x.sum(1) / (~mask).sum(1, keepdim=True)  # average non-pad
        return self.out(pooled)


model = TransformerClassifier(k_clusters, PAD_IDX, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ----------------- 6. evaluation helper -------------------- #
def evaluate(loader, raw_seqs):
    model.eval()
    preds, gts = [], []
    total_loss = 0.0
    with torch.no_grad():
        idx = 0
        for batch in loader:
            x = batch["seq"].to(device)
            mask = x == PAD_IDX
            y = batch["label"].to(device)
            logits = model(x, mask)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            p = logits.argmax(1)
            preds.extend(p.cpu().tolist())
            gts.extend(y.cpu().tolist())
            idx += y.size(0)
    avg_loss = total_loss / len(gts)
    cwa = color_weighted_accuracy(raw_seqs, gts, preds)
    swa = shape_weighted_accuracy(raw_seqs, gts, preds)
    hcs = harmonic_csa(cwa, swa)
    snw = snwa(raw_seqs, gts, preds)
    return {
        "loss": avg_loss,
        "CWA": cwa,
        "SWA": swa,
        "HCSA": hcs,
        "SNWA": snw,
        "preds": preds,
        "gts": gts,
    }


# ----------------- 7. training loop ------------------------ #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
    }
}
best_hcs = -1
patience = 4
since = 0
best_state = None
max_epochs = 20
for epoch in range(1, max_epochs + 1):
    model.train()
    tot_loss = 0
    seen = 0
    for batch in train_loader:
        x = batch["seq"].to(device)
        mask = x == PAD_IDX
        y = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * y.size(0)
        seen += y.size(0)
    train_loss = tot_loss / seen
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))
    val_stats = evaluate(dev_loader, spr["dev"]["sequence"])
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_stats["loss"]))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (
            epoch,
            val_stats["CWA"],
            val_stats["SWA"],
            val_stats["HCSA"],
            val_stats["SNWA"],
        )
    )
    print(
        f'Epoch {epoch}: validation_loss = {val_stats["loss"]:.4f} HCSA={val_stats["HCSA"]:.3f} SNWA={val_stats["SNWA"]:.3f}'
    )
    if val_stats["HCSA"] > best_hcs + 1e-6:
        best_hcs = val_stats["HCSA"]
        best_state = model.state_dict()
        since = 0
    else:
        since += 1
    if since >= patience:
        print("Early stopping.")
        break

if best_state:
    model.load_state_dict(best_state)

# ----------------- 8. final evaluation --------------------- #
dev_final = evaluate(dev_loader, spr["dev"]["sequence"])
test_final = evaluate(test_loader, spr["test"]["sequence"])
experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_final["gts"]
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_final["preds"]
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_final["gts"]

print(
    f'Final Dev  - CWA:{dev_final["CWA"]:.3f} SWA:{dev_final["SWA"]:.3f} HCSA:{dev_final["HCSA"]:.3f} SNWA:{dev_final["SNWA"]:.3f}'
)
print(
    f'Final Test - CWA:{test_final["CWA"]:.3f} SWA:{test_final["SWA"]:.3f} HCSA:{test_final["HCSA"]:.3f} SNWA:{test_final["SNWA"]:.3f}'
)

# ----------------- 9. save everything ---------------------- #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data.")
