import os, pathlib, random, time, json, math, copy, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- reproducibility ---------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- paths ---------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ---------- #
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


# ---------- data loading ---------- #
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

        return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})
    return None


def build_synthetic() -> DatasetDict:
    shapes, colors = list("ABCD"), list("1234")

    def gen_split(n):
        rows = []
        for i in range(n):
            length = random.randint(3, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    def to_dataset(rows):
        return load_dataset(
            "json", data_files={"train": [json.dumps(r) for r in rows]}, split="train"
        )

    return DatasetDict(
        {
            "train": to_dataset(gen_split(600)),
            "dev": to_dataset(gen_split(150)),
            "test": to_dataset(gen_split(150)),
        }
    )


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# ---------- vocabulary / clustering ---------- #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}  # 0 pad
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(token_vecs)
tok2cluster = {tok: int(c) + 1 for tok, c in zip(token_set, kmeans.labels_)}  # 0 pad


# ---------- dataset & dataloader ---------- #
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq_tokens = self.seqs[idx].split()
        shape_ids = [shape2id[t[0]] for t in seq_tokens]
        color_ids = [color2id[t[1]] for t in seq_tokens]
        cluster_ids = [tok2cluster[t] for t in seq_tokens]
        return {
            "shape": shape_ids,
            "color": color_ids,
            "cluster": cluster_ids,
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
train_loader = DataLoader(
    SPRTorchDataset("train"), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset("dev"), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDataset("test"), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------- model ---------- #
class GlyphModel(nn.Module):
    def __init__(self, n_shape, n_color, n_cluster, num_classes, emb_dim=8):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb_dim, padding_idx=0)
        self.cluster_emb = nn.Embedding(n_cluster + 1, emb_dim, padding_idx=0)
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
        return self.ff(pooled)


# ---------- helper: evaluation ---------- #
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, preds, tgts, seqs = 0.0, [], [], []
    for batch in loader:
        shapes, colors, clusters, mask = (
            batch[k].to(device) for k in ["shape", "color", "cluster", "mask"]
        )
        labels = batch["labels"].to(device)
        logits = model(shapes, colors, clusters, mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        p = logits.argmax(1).cpu().tolist()
        preds.extend(p)
        tgts.extend(labels.cpu().tolist())
        seqs.extend(batch["seqs"])
    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        {
            "CWA": color_weighted_accuracy(seqs, tgts, preds),
            "SWA": shape_weighted_accuracy(seqs, tgts, preds),
            "GCWA": glyph_complexity_weighted_accuracy(seqs, tgts, preds),
        },
        preds,
        tgts,
    )


# ---------- hyperparameter sweep ---------- #
learning_rates = [5e-4, 1e-3, 2e-3, 5e-3]
epochs = 5

experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "lrs": learning_rates,
            "losses": {"train": [], "val": []},
            "metrics": {"val": [], "test": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for lr in learning_rates:
    print(f"\n=== Training with learning rate {lr} ===")
    model = GlyphModel(len(shapes), len(colors), n_clusters, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            shapes, colors, clusters, mask = (
                batch[k].to(device) for k in ["shape", "color", "cluster", "mask"]
            )
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(shapes, colors, clusters, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_metrics, _, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| CWA={val_metrics['CWA']:.3f} | SWA={val_metrics['SWA']:.3f} | GCWA={val_metrics['GCWA']:.3f}"
        )

    # record results for this lr (final epoch stats)
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["val"].append(val_metrics)

    # final test evaluation
    test_loss, test_metrics, test_preds, test_tgts = evaluate(model, test_loader)
    print(
        f"Test: loss={test_loss:.4f} | CWA={test_metrics['CWA']:.3f} | "
        f"SWA={test_metrics['SWA']:.3f} | GCWA={test_metrics['GCWA']:.3f}"
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["test"].append(
        test_metrics
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["predictions"].append(test_preds)
    experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"].append(test_tgts)

# ---------- persist ---------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
