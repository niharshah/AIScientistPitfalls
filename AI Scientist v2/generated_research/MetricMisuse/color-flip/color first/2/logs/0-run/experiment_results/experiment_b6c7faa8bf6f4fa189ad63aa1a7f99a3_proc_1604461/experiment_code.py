import os, pathlib, random, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- paths ---------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- global exp dict ---------- #
experiment_data = {"weight_decay": {}}  # will fill per wd value

# ---------- utils ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- data ---------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def try_load_spr(root: pathlib.Path) -> DatasetDict | None:
    if root.exists():

        def _load(csv_name):
            return load_dataset(
                "csv",
                data_files=str(root / csv_name),
                split="train",
                cache_dir=".cache_dsets",
            )

        out = DatasetDict()
        for sp in ["train", "dev", "test"]:
            out[sp] = _load(f"{sp}.csv")
        return out
    return None


def build_synthetic() -> DatasetDict:
    shapes, colors = list("ABCD"), list("1234")

    def gen(n):
        rows = []
        for i in range(n):
            ln = random.randint(3, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(ln)
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    d = DatasetDict()
    for split, n in [("train", 500), ("dev", 100), ("test", 100)]:
        txt = [json.dumps(r) for r in gen(n)]
        d[split] = load_dataset("json", data_files={"train": txt}, split="train")
    return d


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# vocab & clustering (shared across runs)
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
tok2cluster = {
    tok: int(c) + 1
    for tok, c in zip(
        token_set, KMeans(n_clusters, random_state=0).fit(token_vecs).labels_
    )
}


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seq, self.lab = spr[split]["sequence"], spr[split]["label"]

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        seq = self.seq[idx].split()
        return {
            "shape": [shape2id[t[0]] for t in seq],
            "color": [color2id[t[1]] for t in seq],
            "cluster": [tok2cluster[t] for t in seq],
            "label": self.lab[idx],
            "seq_str": self.seq[idx],
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
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "seqs": [b["seq_str"] for b in batch],
    }
    return out


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset("train"), batch_size, True, collate_fn=collate
)
dev_loader = DataLoader(SPRTorchDataset("dev"), batch_size, False, collate_fn=collate)
test_loader = DataLoader(SPRTorchDataset("test"), batch_size, False, collate_fn=collate)


class GlyphModel(nn.Module):
    def __init__(self, n_shape, n_color, n_cluster, n_cls, emb=8):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, emb, padding_idx=0)
        self.ff = nn.Sequential(nn.Linear(emb * 3, 64), nn.ReLU(), nn.Linear(64, n_cls))

    def forward(self, shp, clr, clust, mask):
        e = torch.cat(
            [self.shape_emb(shp), self.color_emb(clr), self.cluster_emb(clust)], dim=-1
        )
        mask = mask.unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1)
        return self.ff(pooled)


def evaluate(mod, loader, criterion):
    mod.eval()
    tot_loss, preds, tgts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for b in loader:
            shp, clr, clu, msk = (
                b["shape"].to(device),
                b["color"].to(device),
                b["cluster"].to(device),
                b["mask"].to(device),
            )
            lbl = b["labels"].to(device)
            logit = mod(shp, clr, clu, msk)
            loss = criterion(logit, lbl)
            tot_loss += loss.item() * lbl.size(0)
            p = logit.argmax(1).cpu().tolist()
            preds += p
            tgts += lbl.cpu().tolist()
            seqs += b["seqs"]
    avg = tot_loss / len(loader.dataset)
    return (
        avg,
        {
            "CWA": color_weighted_accuracy(seqs, tgts, preds),
            "SWA": shape_weighted_accuracy(seqs, tgts, preds),
            "GCWA": glyph_complexity_weighted_accuracy(seqs, tgts, preds),
        },
        preds,
        tgts,
    )


# ---------- sweep ---------- #
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
epochs = 5

for wd in weight_decays:
    key = str(wd)
    print(f"\n=== Training with weight_decay={wd} ===")
    experiment_data["weight_decay"][key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = GlyphModel(len(shapes), len(colors), n_clusters, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for b in train_loader:
            shp, clr, clu, msk = (
                b["shape"].to(device),
                b["color"].to(device),
                b["cluster"].to(device),
                b["mask"].to(device),
            )
            lbl = b["labels"].to(device)
            optimizer.zero_grad()
            logit = model(shp, clr, clu, msk)
            loss = criterion(logit, lbl)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * lbl.size(0)
        train_loss = run_loss / len(train_loader.dataset)

        val_loss, val_metrics, _, _ = evaluate(model, dev_loader, criterion)
        ed = experiment_data["weight_decay"][key]["SPR_BENCH"]
        ed["losses"]["train"].append(train_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["val"].append(val_metrics)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"CWA={val_metrics['CWA']:.3f} | SWA={val_metrics['SWA']:.3f} | GCWA={val_metrics['GCWA']:.3f}"
        )

    # final test
    test_loss, test_metrics, preds, tgts = evaluate(model, test_loader, criterion)
    ed["metrics"]["test"] = test_metrics
    ed["predictions"], ed["ground_truth"] = preds, tgts
    print(
        f"Test: loss={test_loss:.4f} | CWA={test_metrics['CWA']:.3f} | "
        f"SWA={test_metrics['SWA']:.3f} | GCWA={test_metrics['GCWA']:.3f}"
    )

# ---------- save ---------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
