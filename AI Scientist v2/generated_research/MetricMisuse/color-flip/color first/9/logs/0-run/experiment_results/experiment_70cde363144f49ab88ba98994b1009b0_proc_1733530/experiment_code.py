import os, math, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ---------- house-keeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_color_variety(seq: str) -> int:
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ---------- load data ----------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATA", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- glyph clustering ----------
def token_feature(tok: str):
    arr = [ord(c) for c in tok]
    return [arr[0], float(sum(arr[1:])) / max(1, len(arr) - 1)]


tokens = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
feat = np.array([token_feature(t) for t in tokens])
n_clusters = max(8, int(math.sqrt(len(tokens))))
print(f"Clustering {len(tokens)} tokens into {n_clusters}")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(feat)
glyph2cluster = {
    t: int(c) + 1 for t, c in zip(tokens, kmeans.labels_)
}  # 0 reserved for PAD


# ---------- dataset ----------
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [int(l) for l in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [glyph2cluster.get(t, 0) for t in self.seqs[idx].split()]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx]),
            "seq": self.seqs[idx],
        }


def collate(batch):
    lens = [len(b["ids"]) for b in batch]
    max_len = max(lens)
    padded = [
        torch.cat([b["ids"], torch.zeros(max_len - len(b["ids"]), dtype=torch.long)])
        for b in batch
    ]
    return {
        "ids": torch.stack(padded),
        "lens": torch.tensor(lens),
        "label": torch.stack([b["label"] for b in batch]),
        "seq": [b["seq"] for b in batch],
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=512, shuffle=False, collate_fn=collate
)

num_labels = len(set(spr["train"]["label"]))
vocab_size = n_clusters + 1  # plus PAD


# ---------- model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model, nhead, num_layers, n_class):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, n_class)

    def forward(self, x, lens):
        mask = x == 0
        e = self.emb(x)
        e = self.pos(e)
        z = self.enc(e, src_key_padding_mask=mask)
        z = z.masked_fill(mask.unsqueeze(-1), 0.0)
        pooled = z.sum(1) / lens.unsqueeze(-1)
        return self.fc(pooled)


model = TransformerClassifier(vocab_size, 64, 4, 2, num_labels).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()


# ---------- CNA metric ----------
def cluster_normalized_accuracy(seqs, y_true, y_pred):
    cluster_correct = {c: [0, 0] for c in range(1, n_clusters + 1)}
    for s, yt, yp in zip(seqs, y_true, y_pred):
        cl_set = {glyph2cluster.get(t, 0) for t in s.split() if glyph2cluster.get(t, 0)}
        for c in cl_set:
            cluster_correct[c][1] += 1
            if yt == yp:
                cluster_correct[c][0] += 1
    per_acc = [v[0] / v[1] for v in cluster_correct.values() if v[1] > 0]
    return sum(per_acc) / len(per_acc) if per_acc else 0.0


# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- evaluation ----------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, preds, gts, seqs = 0.0, [], [], []
    for batch in loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch_t["ids"], batch_t["lens"])
        loss = criterion(logits, batch_t["label"])
        total_loss += loss.item() * batch_t["label"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch_t["label"].cpu().tolist())
        seqs.extend(batch["seq"])
    avg_loss = total_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cna = cluster_normalized_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    return avg_loss, cwa, swa, cna, hwa, preds, gts, seqs


# ---------- training loop ----------
epochs = 5
for ep in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optim.zero_grad()
        logits = model(batch_t["ids"], batch_t["lens"])
        loss = criterion(logits, batch_t["label"])
        loss.backward()
        optim.step()
        tr_loss += loss.item() * batch_t["label"].size(0)
    tr_loss /= len(train_loader.dataset)

    val_loss, cwa, swa, cna, hwa, *_ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, cwa, swa, cna, hwa))

    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CNA={cna:.3f} HWA={hwa:.3f}"
    )

# ---------- final test ----------
test_loss, cwa, swa, cna, hwa, preds, gts, _ = evaluate(test_loader)
print(
    f"TEST | loss={test_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} CNA={cna:.3f} HWA={hwa:.3f}"
)
experiment_data["SPR_BENCH"]["metrics"]["test"] = (cwa, swa, cna, hwa)
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
