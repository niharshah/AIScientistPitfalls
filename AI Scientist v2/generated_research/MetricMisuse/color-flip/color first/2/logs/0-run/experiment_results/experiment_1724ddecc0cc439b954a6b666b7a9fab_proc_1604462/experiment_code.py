import os, pathlib, random, json, time, warnings

warnings.filterwarnings("ignore")

# ---------------- basic setup ---------------- #
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------- metrics utils --------------- #
def count_color_variety(sequence: str):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str):
    return len(set(tok[0] for tok in sequence.split() if tok))


def cwa(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    return sum([wt if t == pr else 0 for wt, t, pr in zip(w, y, p)]) / max(sum(w), 1)


def swa(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    return sum([wt if t == pr else 0 for wt, t, pr in zip(w, y, p)]) / max(sum(w), 1)


def gcwa(seqs, y, p):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum([wt if t == pr else 0 for wt, t, pr in zip(w, y, p)]) / max(sum(w), 1)


# --------------- data ------------------------ #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def try_load_spr(root: pathlib.Path):
    if root.exists():

        def _load(csv):
            return load_dataset(
                "csv",
                data_files=str(root / csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})
    return None


def build_synth():
    shapes, colors = list("ABCD"), list("1234")

    def gen(n):
        rows = []
        for i in range(n):
            length = random.randint(3, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    d = DatasetDict()
    for sp, n in [("train", 500), ("dev", 100), ("test", 100)]:
        d[sp] = load_dataset(
            "json", data_files={sp: [json.dumps(r) for r in gen(n)]}, split="train"
        )
    return d


spr = try_load_spr(DATA_PATH) or build_synth()
num_classes = len(set(spr["train"]["label"]))

# vocab & clustering
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in sorted(set(all_tokens))], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(token_vecs)
tok2cluster = {
    tok: int(c) + 1 for tok, c in zip(sorted(set(all_tokens)), kmeans.labels_)
}


# dataset class
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx].split()
        sid = [shape2id[t[0]] for t in seq]
        cid = [color2id[t[1]] for t in seq]
        clid = [tok2cluster[t] for t in seq]
        return {
            "shape": sid,
            "color": cid,
            "cluster": clid,
            "label": self.labels[idx],
            "seq_str": self.seqs[idx],
        }


def collate(batch):
    L = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (L - len(b[key])) for b in batch], dtype=torch.long
        )

    shapes, colors, clusters = pad("shape"), pad("color"), pad("cluster")
    mask = (shapes != 0).float()
    labels = torch.tensor([b["label"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {
        "shape": shapes,
        "color": colors,
        "cluster": clusters,
        "mask": mask,
        "labels": labels,
        "seqs": seqs,
    }


batch_size = 128
train_loader = lambda: DataLoader(
    SPRTorchDataset("train"), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset("dev"), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset("test"), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# --------------- model ----------------------- #
class GlyphModel(nn.Module):
    def __init__(
        self, n_shape, n_color, n_cluster, num_classes, emb_dim=8, dropout_rate=0.3
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb_dim, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim * 3, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, shapes, colors, clusters, mask):
        e = torch.cat(
            [
                self.shape_emb(shapes),
                self.color_emb(colors),
                self.cluster_emb(clusters),
            ],
            dim=-1,
        )
        mask = mask.unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1)
        pooled = self.dropout(pooled)
        return self.ff(pooled)


# -------------- training / eval utils ------- #
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    all_preds, all_tgts, all_seqs = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            shp, col, clu, msk = [
                batch[k].to(device) for k in ["shape", "color", "cluster", "mask"]
            ]
            lbl = batch["labels"].to(device)
            logits = model(shp, col, clu, msk)
            loss = criterion(logits, lbl)
            total_loss += loss.item() * lbl.size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(lbl.cpu().tolist())
            all_seqs.extend(batch["seqs"])
    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        {
            "CWA": cwa(all_seqs, all_tgts, all_preds),
            "SWA": swa(all_seqs, all_tgts, all_preds),
            "GCWA": gcwa(all_seqs, all_tgts, all_preds),
        },
        all_preds,
        all_tgts,
    )


# -------------- hyper-parameter sweep -------- #
dropout_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
experiment_data = {"dropout_rate": {}}
best_gcwa, best_rate = -1.0, None

for p in dropout_grid:
    print(f"\n=== training with dropout={p} ===")
    model = GlyphModel(
        len(shapes), len(colors), n_clusters, num_classes, dropout_rate=p
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    edict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, 6):
        model.train()
        running = 0.0
        for batch in train_loader():
            shp, col, clu, msk = [
                batch[k].to(device) for k in ["shape", "color", "cluster", "mask"]
            ]
            lbl = batch["labels"].to(device)
            optim.zero_grad()
            loss = criterion(model(shp, col, clu, msk), lbl)
            loss.backward()
            optim.step()
            running += loss.item() * lbl.size(0)
        train_loss = running / len(spr["train"])
        val_loss, val_m, _, _ = evaluate(model, dev_loader)
        edict["losses"]["train"].append(train_loss)
        edict["losses"]["val"].append(val_loss)
        edict["metrics"]["val"].append(val_m)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"CWA={val_m['CWA']:.3f} SWA={val_m['SWA']:.3f} GCWA={val_m['GCWA']:.3f}"
        )
    # final test with this dropout
    t_loss, t_metrics, t_preds, t_tgts = evaluate(model, test_loader)
    edict["predictions"] = t_preds
    edict["ground_truth"] = t_tgts
    edict["metrics"]["test"] = t_metrics
    experiment_data["dropout_rate"][p] = {"SPR_BENCH": edict}
    if val_m["GCWA"] > best_gcwa:
        best_gcwa = val_m["GCWA"]
        best_rate = p
        best_test = t_metrics

print(f"\nBest dropout={best_rate:.2f} | Dev GCWA={best_gcwa:.3f}")
print(
    f"Test metrics @ best dropout: CWA={best_test['CWA']:.3f} SWA={best_test['SWA']:.3f} GCWA={best_test['GCWA']:.3f}"
)

# -------------- persist ---------------------- #
np.save("experiment_data.npy", experiment_data)
