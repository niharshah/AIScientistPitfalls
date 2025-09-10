import os, pathlib, random, json, math, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# -------------------------- mandatory work dir --------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------ device ----------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- helper metrics ------------------------------ #
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ------------------------------ data ------------------------------------ #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def try_load_spr(root: pathlib.Path):
    if root.exists():

        def _ld(name):
            return load_dataset(
                "csv",
                data_files=str(root / name),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict({sp: _ld(f"{sp}.csv") for sp in ["train", "dev", "test"]})
    return None


def build_synthetic():
    shapes, colors = list("ABCDE"), list("12345")

    def gen(n):
        rows = []
        for i in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    d = DatasetDict()
    for split, n in [("train", 800), ("dev", 200), ("test", 200)]:
        # write to tmp json file list-of-json-lines style
        path = os.path.join(working_dir, f"{split}.jsonl")
        with open(path, "w") as f:
            for r in gen(n):
                f.write(json.dumps(r) + "\n")
        d[split] = load_dataset("json", data_files=path, split="train")
    return d


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()})

# ---------------------------- vocab + clusters --------------------------- #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(6, len(token_vecs) // 2), 40)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs)
tok2cluster = {t: int(c) + 1 for t, c in zip(token_set, kmeans.labels_)}


# ------------------------------ torch ds --------------------------------- #
class SPRTorchDataset(Dataset):
    def __init__(self, split: str):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        return {
            "shape": [shape2id[t[0]] for t in toks],
            "color": [color2id[t[1]] for t in toks],
            "cluster": [tok2cluster[t] for t in toks],
            "label": self.labels[idx],
            "seq_str": self.seqs[idx],
        }


def collate(batch):
    maxlen = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (maxlen - len(b[key])) for b in batch], dtype=torch.long
        )

    out = {
        "shape": pad("shape"),
        "color": pad("color"),
        "cluster": pad("cluster"),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "seqs": [b["seq_str"] for b in batch],
    }
    return out


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset("train"), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset("dev"), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset("test"), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# -------------------------- model definition ----------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
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
        # x: [B,L,D]
        return x + self.pe[:, : x.size(1), :]


class GlyphTransformer(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        n_cluster,
        num_classes,
        d_emb=32,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, d_emb, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, d_emb, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, d_emb, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_emb * 3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb * 3,
            nhead=n_heads,
            dim_feedforward=d_emb * 12,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_emb * 3, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, shapes, colors, clusters, key_padding_mask):
        # shapes/colors/clusters: [B,L]
        x = torch.cat(
            [
                self.shape_emb(shapes),
                self.color_emb(colors),
                self.cluster_emb(clusters),
            ],
            dim=-1,
        )
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        masked = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (x * masked).sum(1) / masked.sum(1)  # mean pool
        return self.classifier(pooled)


# ------------------------ train / eval helpers --------------------------- #
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_tgts, all_seqs = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            shapes, colors, clusters = (
                batch_t["shape"],
                batch_t["color"],
                batch_t["cluster"],
            )
            labels = batch_t["labels"]
            pad_mask = shapes == 0  # bool mask
            logits = model(shapes, colors, clusters, pad_mask)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(labels.cpu().tolist())
            all_seqs.extend(batch["seqs"])
    avg_loss = loss_sum / len(loader.dataset)
    mets = {
        "CWA": color_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "SWA": shape_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "GCWA": glyph_complexity_weighted_accuracy(all_seqs, all_tgts, all_preds),
    }
    return avg_loss, mets, all_preds, all_tgts


# ------------------------- experiment storage ---------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------------------- training loop ------------------------------- #
model_cfgs = [{"d_emb": 32, "layers": 2, "heads": 4}]
for cfg in model_cfgs:
    print("\nCONFIG", cfg)
    model = GlyphTransformer(
        len(shapes),
        len(colors),
        n_clusters,
        num_classes,
        d_emb=cfg["d_emb"],
        n_layers=cfg["layers"],
        n_heads=cfg["heads"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            shapes, colors, clusters = (
                batch_t["shape"],
                batch_t["color"],
                batch_t["cluster"],
            )
            labels = batch_t["labels"]
            pad_mask = shapes == 0
            optimizer.zero_grad()
            logits = model(shapes, colors, clusters, pad_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_mets, _, _ = evaluate(model, dev_loader, criterion)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append({})
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_mets)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | validation_loss = {val_loss:.4f} | "
            f"CWA={val_mets['CWA']:.3f} | SWA={val_mets['SWA']:.3f} | GCWA={val_mets['GCWA']:.3f}"
        )
    # final test
    test_loss, test_mets, test_preds, test_tgts = evaluate(
        model, test_loader, criterion
    )
    experiment_data["SPR_BENCH"]["losses"]["test"] = test_loss
    experiment_data["SPR_BENCH"]["metrics"]["test"] = test_mets
    experiment_data["SPR_BENCH"]["predictions"] = test_preds
    experiment_data["SPR_BENCH"]["ground_truth"] = test_tgts
    print(
        f"TEST: loss={test_loss:.4f} | CWA={test_mets['CWA']:.3f} | SWA={test_mets['SWA']:.3f} | GCWA={test_mets['GCWA']:.3f}"
    )

# ----------------------- persist experiment data ------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
