import os, pathlib, random, json, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# -------------------- misc / reproducibility -------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- helper metrics -------------------- #
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# -------------------- data loading (official or synthetic) -------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def try_load_spr(root: pathlib.Path):
    if root.exists():

        def _load(csv_name):
            return load_dataset(
                "csv",
                data_files=str(root / csv_name),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})
    return None


def build_synthetic():
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

    return DatasetDict(
        {
            sp: load_dataset(
                "json",
                data_files={
                    "train": [json.dumps(r) for r in gen(500 if sp == "train" else 100)]
                },
                split="train",
            )
            for sp in ["train", "dev", "test"]
        }
    )


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# -------------------- vocab & clustering -------------------- #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in sorted(set(all_tokens))], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs)
tok2cluster = {
    tok: int(c) + 1 for tok, c in zip(sorted(set(all_tokens)), kmeans.labels_)
}


# -------------------- torch dataset -------------------- #
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx].split()
        return {
            "shape": [shape2id[t[0]] for t in seq],
            "color": [color2id[t[1]] for t in seq],
            "cluster": [tok2cluster[t] for t in seq],
            "label": self.labels[idx],
            "seq_str": self.seqs[idx],
        }


def collate(batch):
    maxlen = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (maxlen - len(b[key])) for b in batch], dtype=torch.long
        )

    shapes, colors, clusters = pad("shape"), pad("color"), pad("cluster")
    mask = (shapes != 0).float()
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
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


# -------------------- model def -------------------- #
class GlyphModel(nn.Module):
    def __init__(
        self, n_shape, n_color, n_cluster, num_classes, emb_dim=8, hidden_dim=64
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb_dim, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, emb_dim, padding_idx=0)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, s, c, k, mask):
        e = torch.cat(
            [self.shape_emb(s), self.color_emb(c), self.cluster_emb(k)], dim=-1
        )
        mask = mask.unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1)
        return self.ff(pooled)


# -------------------- train / eval helpers -------------------- #
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, all_p, all_t, seqs = 0.0, [], [], []
    with torch.no_grad():
        for b in loader:
            s, c, k = (
                b["shape"].to(device),
                b["color"].to(device),
                b["cluster"].to(device),
            )
            m = b["mask"].to(device)
            labels = b["labels"].to(device)
            logits = model(s, c, k, m)
            loss = criterion(logits, labels)
            tot_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_p.extend(preds)
            all_t.extend(labels.cpu().tolist())
            seqs.extend(b["seqs"])
    l = tot_loss / len(loader.dataset)
    return (
        l,
        {
            "CWA": color_weighted_accuracy(seqs, all_t, all_p),
            "SWA": shape_weighted_accuracy(seqs, all_t, all_p),
            "GCWA": glyph_complexity_weighted_accuracy(seqs, all_t, all_p),
        },
        all_p,
        all_t,
    )


# -------------------- hyperparameter sweep -------------------- #
hidden_dims = [32, 64, 128, 256]
epochs = 5
experiment_data = {
    "hidden_dim": {
        "SPR_BENCH": {
            "metrics": {"train": {}, "val": {}, "test": {}},
            "losses": {"train": {}, "val": {}, "test": {}},
            "predictions": {},
            "ground_truth": {},
        }
    }
}

for hd in hidden_dims:
    print(f"\n----- Training with hidden_dim={hd} -----")
    model = GlyphModel(
        len(shapes), len(colors), n_clusters, num_classes, hidden_dim=hd
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    tr_losses, val_losses, train_metrics, val_metrics = [], [], [], []
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for batch in train_loader():
            s, c, k = (
                batch["shape"].to(device),
                batch["color"].to(device),
                batch["cluster"].to(device),
            )
            m = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(s, c, k, m)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
        train_loss = run_loss / len(spr["train"])
        val_loss, val_metric, _, _ = evaluate(model, dev_loader, criterion)
        tr_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(val_metric)  # storing same type for consistency
        val_metrics.append(val_metric)
        print(
            f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"CWA={val_metric['CWA']:.3f} SWA={val_metric['SWA']:.3f} GCWA={val_metric['GCWA']:.3f}"
        )

    # final test
    test_loss, test_metric, test_preds, test_gts = evaluate(
        model, test_loader, criterion
    )
    print(
        f"TEST  loss={test_loss:.4f}  CWA={test_metric['CWA']:.3f}  "
        f"SWA={test_metric['SWA']:.3f}  GCWA={test_metric['GCWA']:.3f}"
    )

    # store
    ed = experiment_data["hidden_dim"]["SPR_BENCH"]
    ed["losses"]["train"][hd] = tr_losses
    ed["losses"]["val"][hd] = val_losses
    ed["metrics"]["train"][hd] = train_metrics
    ed["metrics"]["val"][hd] = val_metrics
    ed["metrics"]["test"][hd] = test_metric
    ed["losses"]["test"][hd] = test_loss
    ed["predictions"][hd] = test_preds
    ed["ground_truth"][hd] = test_gts

# -------------------- save all -------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll results saved to", os.path.join(working_dir, "experiment_data.npy"))
