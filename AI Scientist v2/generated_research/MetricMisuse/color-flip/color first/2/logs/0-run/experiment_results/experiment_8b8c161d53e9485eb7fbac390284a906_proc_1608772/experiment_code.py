import os, pathlib, json, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# -------- working dir & device ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------- metric helpers ------------------------------------------------------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# -------- load SPR-BENCH or tiny synthetic fallback ---------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr(path: pathlib.Path) -> DatasetDict:
    def _part(fname):
        return load_dataset(
            "csv", data_files=str(path / fname), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {split: _part(f"{split}.csv") for split in ["train", "dev", "test"]}
    )


def tiny_synthetic():
    shapes, colors = list("ABCD"), list("1234")

    def gen(n):
        for i in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            yield {"id": i, "sequence": seq, "label": random.randint(0, 3)}

    d = DatasetDict()
    for split, n in [("train", 800), ("dev", 200), ("test", 200)]:
        tmp = os.path.join(working_dir, f"{split}.jsonl")
        with open(tmp, "w") as f:
            for row in gen(n):
                f.write(json.dumps(row) + "\n")
        d[split] = load_dataset("json", data_files=tmp, split="train")
    return d


spr = load_spr(DATA_PATH) if DATA_PATH.exists() else tiny_synthetic()
num_classes = len(set(spr["train"]["label"]))

# -------- glyph clustering ----------------------------------------------------
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
vecs = np.array([[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float)
n_clusters = min(max(8, len(vecs) // 2), 40)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vecs)
tok2cluster = {tok: int(lbl) + 1 for tok, lbl in zip(token_set, kmeans.labels_)}


# -------- torch Dataset -------------------------------------------------------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.lbls = spr[split]["label"]

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        parts = seq.split()
        shp = [shape2id[t[0]] for t in parts]
        col = [color2id[t[1]] for t in parts]
        clu = [tok2cluster[t] for t in parts]
        return {
            "shape": shp,
            "color": col,
            "cluster": clu,
            "label": self.lbls[idx],
            "seq": seq,
            "shape_var": count_shape_variety(seq),
            "color_var": count_color_variety(seq),
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
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "mask": (pad("shape") != 0).float(),
        "seq": [b["seq"] for b in batch],
        "shape_var": torch.tensor([b["shape_var"] for b in batch], dtype=torch.long),
        "color_var": torch.tensor([b["color_var"] for b in batch], dtype=torch.long),
    }
    return out


batch_size = 256
loaders = {
    split: DataLoader(
        SPRTorch(split),
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate,
    )
    for split in ["train", "dev", "test"]
}


# -------- model ---------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class VarietyAwareTransformer(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        n_cluster,
        num_classes,
        d_model=96,
        nhead=4,
        nlayers=2,
        dropout=0.1,
        max_var=20,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, d_model, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, d_model, padding_idx=0)
        self.shapeVar_emb = nn.Embedding(max_var + 1, d_model)
        self.colorVar_emb = nn.Embedding(max_var + 1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.proj = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, shape, color, cluster, mask, shape_var, color_var):
        bsz = shape.size(0)
        x = self.shape_emb(shape) + self.color_emb(color) + self.cluster_emb(cluster)
        cls = self.cls.repeat(bsz, 1, 1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        attn_mask = torch.cat(
            [torch.ones(bsz, 1, device=mask.device), mask], dim=1
        ).bool()
        x = self.encoder(x, src_key_padding_mask=~attn_mask)
        cls_out = x[:, 0]  # (B,d)
        sv = self.shapeVar_emb(shape_var.clamp(max=20))
        cv = self.colorVar_emb(color_var.clamp(max=20))
        fused = self.proj(torch.cat([cls_out, sv, cv], dim=-1))
        return self.out(fused)


model = VarietyAwareTransformer(len(shapes), len(colors), n_clusters, num_classes).to(
    device
)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# -------- training utilities --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


def evaluate(net, loader):
    net.eval()
    all_p, all_t, all_s = [], [], []
    tot_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = net(
                batch["shape"],
                batch["color"],
                batch["cluster"],
                batch["mask"],
                batch["shape_var"],
                batch["color_var"],
            )
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_p.extend(preds)
            all_t.extend(batch["label"].cpu().tolist())
            all_s.extend(batch["seq"])
    avg = tot_loss / len(loader.dataset)
    mets = {
        "CWA": color_weighted_accuracy(all_s, all_t, all_p),
        "SWA": shape_weighted_accuracy(all_s, all_t, all_p),
        "GCWA": glyph_complexity_weighted_accuracy(all_s, all_t, all_p),
    }
    return avg, mets, all_p, all_t


# -------- experiment logging --------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "variety_transformer": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# -------- training loop -------------------------------------------------------
epochs = 10
for ep in range(1, epochs + 1):
    model.train()
    run_loss = 0.0
    for batch in loaders["train"]:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch["shape"],
            batch["color"],
            batch["cluster"],
            batch["mask"],
            batch["shape_var"],
            batch["color_var"],
        )
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["label"].size(0)
    tr_loss = run_loss / len(loaders["train"].dataset)
    val_loss, val_mets, _, _ = evaluate(model, loaders["dev"])
    experiment_data["SPR_BENCH"]["variety_transformer"]["losses"]["train"].append(
        tr_loss
    )
    experiment_data["SPR_BENCH"]["variety_transformer"]["losses"]["val"].append(
        val_loss
    )
    experiment_data["SPR_BENCH"]["variety_transformer"]["metrics"]["train"].append({})
    experiment_data["SPR_BENCH"]["variety_transformer"]["metrics"]["val"].append(
        val_mets
    )
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | "
        f'CWA={val_mets["CWA"]:.3f} | SWA={val_mets["SWA"]:.3f} | GCWA={val_mets["GCWA"]:.3f}'
    )
    scheduler.step()

# -------- final test ----------------------------------------------------------
test_loss, test_mets, test_preds, test_gt = evaluate(model, loaders["test"])
print(
    f'Test: loss={test_loss:.4f} | CWA={test_mets["CWA"]:.3f} | '
    f'SWA={test_mets["SWA"]:.3f} | GCWA={test_mets["GCWA"]:.3f}'
)
experiment_data["SPR_BENCH"]["variety_transformer"]["losses"]["test"] = test_loss
experiment_data["SPR_BENCH"]["variety_transformer"]["metrics"]["test"] = test_mets
experiment_data["SPR_BENCH"]["variety_transformer"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["variety_transformer"]["ground_truth"] = test_gt

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
