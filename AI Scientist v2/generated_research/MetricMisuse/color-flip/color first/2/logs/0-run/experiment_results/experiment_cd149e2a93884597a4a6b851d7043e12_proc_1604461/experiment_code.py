import os, pathlib, random, json, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- experiment dictionary ---------- #
experiment_data = {
    "batch_size": {"SPR_BENCH": {}}  # will hold results keyed by batch size
}

# ---------- device ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- metrics utilities ---------- #
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


def try_load_spr(root: pathlib.Path) -> DatasetDict:
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
            seq = " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(3, 9))
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    d = DatasetDict()
    d["train"] = load_dataset(
        "json",
        data_files={"train": [json.dumps(r) for r in gen_split(500)]},
        split="train",
    )
    d["dev"] = load_dataset(
        "json",
        data_files={"train": [json.dumps(r) for r in gen_split(100)]},
        split="train",
    )
    d["test"] = load_dataset(
        "json",
        data_files={"train": [json.dumps(r) for r in gen_split(100)]},
        split="train",
    )
    return d


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# ---------- vocab & clustering ---------- #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}  # 0 reserved for PAD
color2id = {c: i + 1 for i, c in enumerate(colors)}

token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs)
tok2cluster = {tok: int(c) + 1 for tok, c in zip(token_set, kmeans.labels_)}  # 0 pad


# ---------- dataset class ---------- #
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = spr[split]["sequence"], spr[split]["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx].split()
        shape_ids = [shape2id[t[0]] for t in seq]
        color_ids = [color2id[t[1]] for t in seq]
        cluster_ids = [tok2cluster[t] for t in seq]
        return {
            "shape": shape_ids,
            "color": color_ids,
            "cluster": cluster_ids,
            "label": self.labels[idx],
            "seq_str": self.seqs[idx],
        }


def collate(batch):
    maxlen = max(len(x["shape"]) for x in batch)

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
            -1,
        )
        mask = mask.unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1)
        return self.ff(pooled)


# ---------- evaluation helper ---------- #
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_tgts, all_seqs = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            shapes = batch["shape"].to(device)
            colors = batch["color"].to(device)
            clusters = batch["cluster"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(shapes, colors, clusters, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(labels.cpu().tolist())
            all_seqs.extend(batch["seqs"])
    avg_loss = total_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(all_seqs, all_tgts, all_preds)
    swa = shape_weighted_accuracy(all_seqs, all_tgts, all_preds)
    gcwa = glyph_complexity_weighted_accuracy(all_seqs, all_tgts, all_preds)
    return avg_loss, {"CWA": cwa, "SWA": swa, "GCWA": gcwa}, all_preds, all_tgts


# ---------- hyper-parameter sweep ---------- #
batch_sizes = [32, 64, 128, 256]
epochs = 5

for bs in batch_sizes:
    print(f"\n=== Training with batch size {bs} ===")
    # data loaders
    train_loader = DataLoader(
        SPRTorchDataset("train"), batch_size=bs, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        SPRTorchDataset("dev"), batch_size=bs, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        SPRTorchDataset("test"), batch_size=bs, shuffle=False, collate_fn=collate
    )

    # model / optim
    model = GlyphModel(len(shapes), len(colors), n_clusters, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage dict
    exp_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            shapes = batch["shape"].to(device)
            colors = batch["color"].to(device)
            clusters = batch["cluster"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(shapes, colors, clusters, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_metrics, _, _ = evaluate(model, dev_loader, criterion)

        exp_dict["losses"]["train"].append(train_loss)
        exp_dict["losses"]["val"].append(val_loss)
        exp_dict["metrics"]["val"].append(val_metrics)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"CWA={val_metrics['CWA']:.3f} | SWA={val_metrics['SWA']:.3f} | GCWA={val_metrics['GCWA']:.3f}"
        )

    # final test
    test_loss, test_metrics, preds, tgts = evaluate(model, test_loader, criterion)
    print(
        f"Test: loss={test_loss:.4f} | CWA={test_metrics['CWA']:.3f} | "
        f"SWA={test_metrics['SWA']:.3f} | GCWA={test_metrics['GCWA']:.3f}"
    )

    exp_dict["predictions"] = preds
    exp_dict["ground_truth"] = tgts
    exp_dict["metrics"]["test"] = test_metrics

    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = exp_dict

    # clean up
    del model
    torch.cuda.empty_cache()

# ---------- persist all results ---------- #
np.save("experiment_data.npy", experiment_data)
print("\nSaved experiment data to experiment_data.npy")
