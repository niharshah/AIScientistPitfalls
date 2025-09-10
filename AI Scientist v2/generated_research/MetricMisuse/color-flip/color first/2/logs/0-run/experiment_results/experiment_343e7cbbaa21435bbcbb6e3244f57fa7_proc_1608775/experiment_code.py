import os, pathlib, json, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------------- directory & device ---------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- metrics --------------------------- #
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, pr in zip(w, y, p) if t == pr) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, pr in zip(w, y, p) if t == pr) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y, p):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, pr in zip(w, y, p) if t == pr) / max(sum(w), 1)


# ---------------- data loading ---------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(path):
    def _ld(fn):
        return load_dataset(
            "csv", data_files=str(path / fn), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _ld(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def build_tiny_synthetic():
    shapes, colors = list("ABCDE"), list("12345")

    def gen(n):
        rows = []
        for i in range(n):
            ln = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(ln)
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    dd = DatasetDict()
    for sp, n in [("train", 800), ("dev", 200), ("test", 200)]:
        fp = os.path.join(working_dir, f"{sp}.jsonl")
        with open(fp, "w") as f:
            [f.write(json.dumps(r) + "\n") for r in gen(n)]
        dd[sp] = load_dataset("json", data_files=fp, split="train")
    return dd


spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_tiny_synthetic()
num_classes = len(set(spr["train"]["label"]))

# ---------------- glyph clustering ------------------ #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
vecs = np.array([[shape2id[t[0]], color2id[t[1]]] for t in token_set], float)
n_clusters = min(max(12, len(vecs) // 2), 50)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vecs)
tok2cluster = {tok: int(label) + 1 for tok, label in zip(token_set, kmeans.labels_)}


# ---------------- torch dataset --------------------- #
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.lbls = spr[split]["label"]

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, idx):
        seq = self.seqs[idx].split()
        return {
            "shape": [shape2id[t[0]] for t in seq],
            "color": [color2id[t[1]] for t in seq],
            "cluster": [tok2cluster[t] for t in seq],
            "label": self.lbls[idx],
            "seq": " ".join(seq),
        }


def collate_fn(batch):
    L = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (L - len(b[key])) for b in batch], dtype=torch.long
        )

    shape = pad("shape")
    color = pad("color")
    cluster = pad("cluster")
    mask = (shape != 0).float()
    return {
        "shape": shape,
        "color": color,
        "cluster": cluster,
        "label": torch.tensor([b["label"] for b in batch]),
        "mask": mask,
        "seq": [b["seq"] for b in batch],
    }


bs = 256
train_loader = DataLoader(
    SPRTorch("train"), batch_size=bs, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorch("dev"), batch_size=bs, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorch("test"), batch_size=bs, shuffle=False, collate_fn=collate_fn
)


# ---------------- model ----------------------------- #
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


class HybridTransformerCNN(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        n_cluster,
        num_classes,
        d_model=64,
        nhead=4,
        nlayers=2,
        kernels=(2, 3, 4, 5),
        dropout=0.1,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, d_model, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding=0) for k in kernels]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model * (1 + len(kernels))),
            nn.Linear(d_model * (1 + len(kernels)), num_classes),
        )

    def forward(self, shape, color, cluster, mask):
        tok = self.shape_emb(shape) + self.color_emb(color) + self.cluster_emb(cluster)
        batch = tok.size(0)
        x = torch.cat([self.cls.repeat(batch, 1, 1), tok], 1)
        x = self.pos(x)
        attn_mask = torch.cat(
            [torch.ones(batch, 1, device=mask.device), mask], 1
        ).bool()
        z = self.transformer(x, src_key_padding_mask=~attn_mask)[:, 0]  # CLS
        # CNN branch (ignore CLS position)
        embs = tok.transpose(1, 2)  # B x C x L
        conv_feats = [
            torch.max(torch.relu(conv(embs)), dim=2)[0] for conv in self.convs
        ]
        feats = torch.cat([z] + conv_feats, dim=1)
        feats = self.dropout(feats)
        return self.fc(feats)


model = HybridTransformerCNN(len(shapes), len(colors), n_clusters, num_classes).to(
    device
)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ---------------- training -------------------------- #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

experiment_data = {
    "SPR_BENCH": {
        "transformer_cnn": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


def evaluate(net, loader):
    net.eval()
    tot_loss = 0
    preds = []
    tgts = []
    seqs = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = net(
                batch["shape"], batch["color"], batch["cluster"], batch["mask"]
            )
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            ps = logits.argmax(1).cpu().tolist()
            preds.extend(ps)
            tgts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    avg = tot_loss / len(loader.dataset)
    m = {
        "CWA": color_weighted_accuracy(seqs, tgts, preds),
        "SWA": shape_weighted_accuracy(seqs, tgts, preds),
        "GCWA": glyph_complexity_weighted_accuracy(seqs, tgts, preds),
    }
    return avg, m, preds, tgts


epochs = 12
for ep in range(1, epochs + 1):
    model.train()
    run_loss = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logit = model(batch["shape"], batch["color"], batch["cluster"], batch["mask"])
        loss = criterion(logit, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run_loss += loss.item() * batch["label"].size(0)
    tr_loss = run_loss / len(train_loader.dataset)
    val_loss, val_metrics, _, _ = evaluate(model, dev_loader)
    experiment_data["SPR_BENCH"]["transformer_cnn"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["transformer_cnn"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["transformer_cnn"]["metrics"]["val"].append(
        val_metrics
    )
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | "
        f"CWA={val_metrics['CWA']:.3f} | SWA={val_metrics['SWA']:.3f} | GCWA={val_metrics['GCWA']:.3f}"
    )
    scheduler.step()

# ---------------- test ------------------------------ #
test_loss, test_metrics, test_preds, test_gt = evaluate(model, test_loader)
print(
    f"Test: loss={test_loss:.4f} | CWA={test_metrics['CWA']:.3f} | "
    f"SWA={test_metrics['SWA']:.3f} | GCWA={test_metrics['GCWA']:.3f}"
)

experiment_data["SPR_BENCH"]["transformer_cnn"]["losses"]["test"] = test_loss
experiment_data["SPR_BENCH"]["transformer_cnn"]["metrics"]["test"] = test_metrics
experiment_data["SPR_BENCH"]["transformer_cnn"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["transformer_cnn"]["ground_truth"] = test_gt

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
