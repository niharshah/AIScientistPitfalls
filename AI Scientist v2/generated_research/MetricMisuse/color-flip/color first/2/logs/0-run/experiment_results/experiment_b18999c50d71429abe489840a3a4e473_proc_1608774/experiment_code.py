import os, pathlib, json, random, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# --------------------------- work dir  &  device --------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------ metrics ----------------------------------- #
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


# -------------------------- dataset loading -------------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")  # original path


def load_spr_bench(path: pathlib.Path):
    def _ld(fname):
        return load_dataset(
            "csv", data_files=str(path / fname), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _ld(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def build_tiny_synthetic():
    shapes, colors = list("ABCD"), list("1234")

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
        tmpfile = os.path.join(working_dir, f"{split}.jsonl")
        with open(tmpfile, "w") as f:
            for row in gen(n):
                f.write(json.dumps(row) + "\n")
        d[split] = load_dataset("json", data_files=tmpfile, split="train")
    return d


spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_tiny_synthetic()
num_classes = len(set(spr["train"]["label"]))

# --------------------------- clustering glyphs ----------------------------- #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
vecs = np.array([[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float)
n_clusters = min(max(8, len(vecs) // 2), 40)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vecs)
tok2cluster = {tok: int(label) + 1 for tok, label in zip(token_set, kmeans.labels_)}


# ------------------------- torch dataset & loader -------------------------- #
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.lbls = spr[split]["label"]

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, idx):
        seq_str = self.seqs[idx]
        parts = seq_str.split()
        shape = [shape2id[t[0]] for t in parts]
        color = [color2id[t[1]] for t in parts]
        cluster = [tok2cluster[t] for t in parts]
        return {
            "shape": shape,
            "color": color,
            "cluster": cluster,
            "label": self.lbls[idx],
            "seq": seq_str,
        }


def collate_fn(batch):
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
    }
    return out


batch_size = 256
train_loader = DataLoader(
    SPRTorch("train"), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorch("dev"), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorch("test"), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


# ------------------------------ model -------------------------------------- #
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


class SPRTransformer(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        n_cluster,
        num_classes,
        d_model=64,
        nhead=4,
        nlayers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, d_model, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))  # learnable CLS
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, shape, color, cluster, mask):
        batch_size = shape.size(0)
        tok_emb = (
            self.shape_emb(shape) + self.color_emb(color) + self.cluster_emb(cluster)
        )
        cls_token = self.cls.repeat(batch_size, 1, 1)
        x = torch.cat([cls_token, tok_emb], dim=1)
        pos_x = self.pos_enc(x)
        attn_mask = torch.cat(
            [torch.ones(batch_size, 1, device=mask.device), mask], dim=1
        ).bool()
        out = self.transformer(pos_x, src_key_padding_mask=~attn_mask)
        cls_out = out[:, 0]
        return self.out(cls_out)


model = SPRTransformer(len(shapes), len(colors), n_clusters, num_classes).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ---------------------------- training setup -------------------------------- #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


def evaluate(net, loader):
    net.eval()
    all_preds, all_tgts, all_seqs = [], [], []
    tot_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = net(b["shape"], b["color"], b["cluster"], b["mask"])
            loss = criterion(logits, b["label"])
            tot_loss += loss.item() * b["label"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(b["label"].cpu().tolist())
            all_seqs.extend(batch["seq"])
    avg_loss = tot_loss / len(loader.dataset)
    metrics = {
        "CWA": color_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "SWA": shape_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "GCWA": glyph_complexity_weighted_accuracy(all_seqs, all_tgts, all_preds),
    }
    return avg_loss, metrics, all_preds, all_tgts


# --------------------------- experiment log dict --------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "transformer_baseline": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ----------------------------- training loop ------------------------------- #
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    run_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["shape"], batch["color"], batch["cluster"], batch["mask"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["label"].size(0)
    train_loss = run_loss / len(train_loader.dataset)
    val_loss, val_metrics, _, _ = evaluate(model, dev_loader)
    experiment_data["SPR_BENCH"]["transformer_baseline"]["losses"]["train"].append(
        train_loss
    )
    experiment_data["SPR_BENCH"]["transformer_baseline"]["losses"]["val"].append(
        val_loss
    )
    experiment_data["SPR_BENCH"]["transformer_baseline"]["metrics"]["train"].append({})
    experiment_data["SPR_BENCH"]["transformer_baseline"]["metrics"]["val"].append(
        val_metrics
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f'CWA={val_metrics["CWA"]:.3f} | SWA={val_metrics["SWA"]:.3f} | GCWA={val_metrics["GCWA"]:.3f}'
    )
    scheduler.step()

# ------------------------------- final test -------------------------------- #
test_loss, test_metrics, test_preds, test_gt = evaluate(model, test_loader)
print(
    f'Test: loss={test_loss:.4f} | CWA={test_metrics["CWA"]:.3f} | '
    f'SWA={test_metrics["SWA"]:.3f} | GCWA={test_metrics["GCWA"]:.3f}'
)

experiment_data["SPR_BENCH"]["transformer_baseline"]["losses"]["test"] = test_loss
experiment_data["SPR_BENCH"]["transformer_baseline"]["metrics"]["test"] = test_metrics
experiment_data["SPR_BENCH"]["transformer_baseline"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["transformer_baseline"]["ground_truth"] = test_gt

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
