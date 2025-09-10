import os, pathlib, random, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# -------------------- paths / dirs -------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- device -------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- metrics -------------------- #
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


# -------------------- data loading -------------------- #
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

        d = DatasetDict()
        for sp in ["train", "dev", "test"]:
            d[sp] = _load(f"{sp}.csv")
        return d
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
        # save to tmp json lines file (datasets can't load from list of dict easily without disk)
        tmp_path = os.path.join(working_dir, f"tmp_{time.time_ns()}.jsonl")
        with open(tmp_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        ds = load_dataset("json", data_files=tmp_path, split="train")
        os.remove(tmp_path)
        return ds

    return DatasetDict(
        {"train": gen_split(600), "dev": gen_split(120), "test": gen_split(120)}
    )


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# -------------------- vocab & clustering -------------------- #
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}  # 0 pad
color2id = {c: i + 1 for i, c in enumerate(colors)}  # 0 pad

token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs)
tok2cluster = {tok: int(c) + 1 for tok, c in zip(token_set, kmeans.labels_)}  # 0 pad


# -------------------- dataset / dataloader -------------------- #
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

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


# -------------------- model -------------------- #
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


# -------------------- evaluation helper -------------------- #
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_tgts, all_seqs = 0.0, [], [], []
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


# -------------------- hyperparameter sweep -------------------- #
embedding_grid = [4, 8, 16, 32]
epochs = 5

experiment_data = {"embedding_dim": {"SPR_BENCH": {}}}

for emb_dim in embedding_grid:
    print(f"\n===== Training with embedding_dim={emb_dim} =====")
    model = GlyphModel(
        len(shapes), len(colors), n_clusters, num_classes, emb_dim=emb_dim
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # training loop
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
        exp_entry["losses"]["train"].append(train_loss)
        exp_entry["losses"]["val"].append(val_loss)
        exp_entry["metrics"]["val"].append(val_metrics)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| CWA={val_metrics['CWA']:.3f} | SWA={val_metrics['SWA']:.3f} | GCWA={val_metrics['GCWA']:.3f}"
        )
    # final test evaluation
    test_loss, test_metrics, test_preds, test_tgts = evaluate(
        model, test_loader, criterion
    )
    exp_entry["metrics"]["test"] = test_metrics
    exp_entry["losses"]["test"] = test_loss
    exp_entry["predictions"] = test_preds
    exp_entry["ground_truth"] = test_tgts
    print(
        f"Test | loss={test_loss:.4f} | CWA={test_metrics['CWA']:.3f} | "
        f"SWA={test_metrics['SWA']:.3f} | GCWA={test_metrics['GCWA']:.3f}"
    )
    experiment_data["embedding_dim"]["SPR_BENCH"][f"dim_{emb_dim}"] = exp_entry

# -------------------- persist -------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
