import os, pathlib, json, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers ----------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def gcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------- data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr(path: pathlib.Path):
    def _ld(fname):
        return load_dataset(
            "csv", data_files=str(path / fname), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _ld(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def build_synthetic():
    shapes, colors = list("ABCD"), list("1234")

    def gen(n):
        for i in range(n):
            ln = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(ln)
            )
            yield {"id": i, "sequence": seq, "label": random.randint(0, 3)}

    d = DatasetDict()
    for sp, n in [("train", 800), ("dev", 200), ("test", 200)]:
        tmp = os.path.join(working_dir, f"{sp}.jsonl")
        with open(tmp, "w") as f:
            for row in gen(n):
                f.write(json.dumps(row) + "\n")
        d[sp] = load_dataset("json", data_files=tmp, split="train")
    return d


spr = load_spr(DATA_PATH) if DATA_PATH.exists() else build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# glyph vocab & clustering
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
vecs = np.array([[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float)
n_clusters = min(max(8, len(vecs) // 2), 40)
tok2cluster = {
    tok: int(l) + 1
    for tok, l in zip(token_set, KMeans(n_clusters, random_state=0).fit(vecs).labels_)
}

# counts range
max_color_var = max(count_color_variety(s) for s in spr["train"]["sequence"]) + 1
max_shape_var = max(count_shape_variety(s) for s in spr["train"]["sequence"]) + 1


# ---------- torch Dataset ----------
class SPRTorch(Dataset):
    def __init__(self, split: str):
        self.seq = spr[split]["sequence"]
        self.lbl = spr[split]["label"]

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        s = self.seq[idx]
        parts = s.split()
        return {
            "shape": [shape2id[t[0]] for t in parts],
            "color": [color2id[t[1]] for t in parts],
            "cluster": [tok2cluster[t] for t in parts],
            "label": self.lbl[idx],
            "c_cnt": count_color_variety(s),
            "s_cnt": count_shape_variety(s),
            "seq": s,
        }


def collate(batch):
    L = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (L - len(b[key])) for b in batch], dtype=torch.long
        )

    out = {
        "shape": pad("shape"),
        "color": pad("color"),
        "cluster": pad("cluster"),
        "mask": (pad("shape") != 0).float(),
        "label": torch.tensor([b["label"] for b in batch]),
        "c_cnt": torch.tensor([b["c_cnt"] for b in batch]),
        "s_cnt": torch.tensor([b["s_cnt"] for b in batch]),
        "seq": [b["seq"] for b in batch],
    }
    return out


bs = 256
train_loader = DataLoader(
    SPRTorch("train"), batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch("dev"), batch_size=bs, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch("test"), batch_size=bs, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class PosEnc(nn.Module):
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


class SPRMultiTask(nn.Module):
    def __init__(
        self,
        n_shape,
        n_color,
        n_cluster,
        num_cls,
        d=64,
        nhead=4,
        layers=2,
        drop=0.1,
        max_c=max_color_var,
        max_s=max_shape_var,
    ):
        super().__init__()
        self.shape_e = nn.Embedding(n_shape + 1, d, padding_idx=0)
        self.color_e = nn.Embedding(n_color + 1, d, padding_idx=0)
        self.cluster_e = nn.Embedding(n_cluster + 1, d, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = PosEnc(d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dropout=drop, batch_first=True
        )
        self.trf = nn.TransformerEncoder(enc, num_layers=layers)
        self.main_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_cls))
        self.col_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, max_c))
        self.shp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, max_s))

    def forward(self, shape, color, cluster, mask):
        b = shape.size(0)
        tok = self.shape_e(shape) + self.color_e(color) + self.cluster_e(cluster)
        x = torch.cat([self.cls.repeat(b, 1, 1), tok], dim=1)
        x = self.pos(x)
        attn_mask = torch.cat(
            [torch.ones(b, 1, device=mask.device), mask], dim=1
        ).bool()
        h = self.trf(x, src_key_padding_mask=~attn_mask)[:, 0]
        return self.main_head(h), self.col_head(h), self.shp_head(h)


model = SPRMultiTask(len(shapes), len(colors), n_clusters, num_classes).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ---------- training setup ----------
ce = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
aux_weight = 0.2

# ---------- experiment log ----------
experiment_data = {
    "SPR_BENCH": {
        "multitask_cluster": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ---------- evaluation ----------
def evaluate(net, loader):
    net.eval()
    preds, tgts, seqs = [], [], []
    loss_sum = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out, out_c, out_s = net(
                batch["shape"], batch["color"], batch["cluster"], batch["mask"]
            )
            loss = ce(out, batch["label"]) + aux_weight * (
                ce(out_c, batch["c_cnt"]) + ce(out_s, batch["s_cnt"])
            )
            loss_sum += loss.item() * batch["label"].size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            tgts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    avg = loss_sum / len(loader.dataset)
    return (
        avg,
        {
            "CWA": cwa(seqs, tgts, preds),
            "SWA": swa(seqs, tgts, preds),
            "GCWA": gcwa(seqs, tgts, preds),
        },
        preds,
        tgts,
    )


# ---------- training loop ----------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    run_loss = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt.zero_grad()
        out, out_c, out_s = model(
            batch["shape"], batch["color"], batch["cluster"], batch["mask"]
        )
        loss_main = ce(out, batch["label"])
        loss_aux = ce(out_c, batch["c_cnt"]) + ce(out_s, batch["s_cnt"])
        loss = loss_main + aux_weight * loss_aux
        loss.backward()
        opt.step()
        run_loss += loss.item() * batch["label"].size(0)
    train_loss = run_loss / len(train_loader.dataset)
    val_loss, val_met, _, _ = evaluate(model, dev_loader)
    experiment_data["SPR_BENCH"]["multitask_cluster"]["losses"]["train"].append(
        train_loss
    )
    experiment_data["SPR_BENCH"]["multitask_cluster"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["multitask_cluster"]["metrics"]["val"].append(val_met)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"CWA={val_met['CWA']:.3f} | SWA={val_met['SWA']:.3f} | GCWA={val_met['GCWA']:.3f}"
    )
    sched.step()

# ---------- final test ----------
test_loss, test_met, test_pred, test_gt = evaluate(model, test_loader)
print(
    f'Test: loss={test_loss:.4f} | CWA={test_met["CWA"]:.3f} | '
    f'SWA={test_met["SWA"]:.3f} | GCWA={test_met["GCWA"]:.3f}'
)

ed = experiment_data["SPR_BENCH"]["multitask_cluster"]
ed["losses"]["test"] = test_loss
ed["metrics"]["test"] = test_met
ed["predictions"] = test_pred
ed["ground_truth"] = test_gt
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
