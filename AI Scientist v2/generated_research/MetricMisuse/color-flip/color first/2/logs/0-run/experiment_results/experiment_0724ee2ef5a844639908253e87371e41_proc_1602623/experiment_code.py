import os, random, json, pathlib, time
import numpy as np
import torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- paths & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- metric helpers ----------
def count_color_variety(seq):  # e.g. A1 B2 C3 -> 3
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.strip().split() if tok})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------- dataset (try official else synthetic) ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def try_load_spr(root: pathlib.Path):
    if root.exists():
        from datasets import load_dataset

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

    def gen(n):  # produce list[dict]
        rows = []
        for i in range(n):
            length = random.randint(3, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            rows.append({"id": i, "sequence": seq, "label": random.randint(0, 3)})
        return rows

    d = DatasetDict()
    for split, n in [("train", 500), ("dev", 100), ("test", 100)]:
        d[split] = HFDataset.from_list(gen(n))
    return d


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# ---------- vocab & token clustering ----------
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}  # 0 reserved for PAD
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in sorted(set(all_tokens))], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs)
tok2cluster = {
    tok: int(c) + 1 for tok, c in zip(sorted(set(all_tokens)), kmeans.labels_)
}  # 0 pad


# ---------- torch Dataset ----------
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


# ---------- model ----------
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
        pooled = (e * mask).sum(1) / (mask.sum(1) + 1e-9)
        return self.ff(pooled)


# ---------- training / evaluation helpers ----------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_tgts, all_seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            shapes, colors, clusters, mask = [
                batch[k].to(device) for k in ["shape", "color", "cluster", "mask"]
            ]
            labels = batch["labels"].to(device)
            logits = model(shapes, colors, clusters, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(labels.cpu().tolist())
            all_seqs.extend(batch["seqs"])
    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        {
            "CWA": color_weighted_accuracy(all_seqs, all_tgts, all_preds),
            "SWA": shape_weighted_accuracy(all_seqs, all_tgts, all_preds),
            "GCWA": glyph_complexity_weighted_accuracy(all_seqs, all_tgts, all_preds),
        },
        all_preds,
        all_tgts,
    )


# ---------- hyper-parameter tuning ----------
batch_sizes = [32, 64, 128, 256]
epochs = 5
experiment_data = {"batch_size": {"SPR_BENCH": {}}}

for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    # loaders
    train_loader = DataLoader(
        SPRTorchDataset("train"), batch_size=bs, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        SPRTorchDataset("dev"), batch_size=bs, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        SPRTorchDataset("test"), batch_size=bs, shuffle=False, collate_fn=collate
    )
    # model, opt
    model = GlyphModel(len(shapes), len(colors), n_clusters, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # storage
    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            shapes, colors, clusters, mask = [
                batch[k].to(device) for k in ["shape", "color", "cluster", "mask"]
            ]
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(shapes, colors, clusters, mask), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
        tr_loss = running / len(train_loader.dataset)
        val_loss, val_metrics, _, _ = evaluate(model, dev_loader, criterion)
        run_data["losses"]["train"].append(tr_loss)
        run_data["losses"]["val"].append(val_loss)
        run_data["metrics"]["val"].append(val_metrics)
        print(
            f"Epoch {ep} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
            f"CWA {val_metrics['CWA']:.3f} | SWA {val_metrics['SWA']:.3f} | GCWA {val_metrics['GCWA']:.3f}"
        )
    # final test
    test_loss, test_metrics, test_preds, test_tgts = evaluate(
        model, test_loader, criterion
    )
    run_data["predictions"] = test_preds
    run_data["ground_truth"] = test_tgts
    run_data["metrics"]["test"] = test_metrics
    print(
        f"Test | loss {test_loss:.4f} | CWA {test_metrics['CWA']:.3f} | "
        f"SWA {test_metrics['SWA']:.3f} | GCWA {test_metrics['GCWA']:.3f}"
    )
    experiment_data["batch_size"]["SPR_BENCH"][bs] = run_data
    del model
    torch.cuda.empty_cache()

# ---------- persist ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
