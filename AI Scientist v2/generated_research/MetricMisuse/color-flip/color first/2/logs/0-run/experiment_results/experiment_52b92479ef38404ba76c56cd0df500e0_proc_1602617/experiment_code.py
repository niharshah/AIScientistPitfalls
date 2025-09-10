# num_epochs hyper-parameter tuning – single-file runnable script
import os, pathlib, random, json, copy, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ------------------------------------------------------------------------- #
# misc helpers & paths
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------------- #
# metric utilities
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


# ------------------------------------------------------------------------- #
# data loading (uses official SPR_BENCH if present, otherwise synthetic)
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

        return DatasetDict({sp: _load(f"{sp}.csv") for sp in ("train", "dev", "test")})
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

    d = DatasetDict()
    for sp, n in [("train", 500), ("dev", 100), ("test", 100)]:
        json_strs = [json.dumps(r) for r in gen_split(n)]
        d[sp] = load_dataset("json", data_files={"train": json_strs}, split="train")
    return d


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# ------------------------------------------------------------------------- #
# vocabularies & token clustering
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}  # 0 = pad
color2id = {c: i + 1 for i, c in enumerate(colors)}

token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs)
tok2cluster = {tok: int(c) + 1 for tok, c in zip(token_set, kmeans.labels_)}  # 0 pad


# ------------------------------------------------------------------------- #
# torch dataset
class SPRTorchDataset(Dataset):
    def __init__(self, split: str):
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


# common loaders
batch_size = 128
train_loader_full = DataLoader(
    SPRTorchDataset("train"), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader_full = DataLoader(
    SPRTorchDataset("dev"), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader_full = DataLoader(
    SPRTorchDataset("test"), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ------------------------------------------------------------------------- #
# model
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


# ------------------------------------------------------------------------- #
# evaluation helper
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, preds, tgts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            shapes = batch["shape"].to(device)
            colors = batch["color"].to(device)
            clusters = batch["cluster"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(shapes, colors, clusters, mask)
            loss = criterion(logits, labels)
            tot_loss += loss.item() * labels.size(0)
            pred = logits.argmax(1).cpu().tolist()
            preds.extend(pred)
            tgts.extend(labels.cpu().tolist())
            seqs.extend(batch["seqs"])
    avg_loss = tot_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, tgts, preds)
    swa = shape_weighted_accuracy(seqs, tgts, preds)
    gcwa = glyph_complexity_weighted_accuracy(seqs, tgts, preds)
    return avg_loss, {"CWA": cwa, "SWA": swa, "GCWA": gcwa}, preds, tgts


# ------------------------------------------------------------------------- #
# experiment data container
experiment_data = {"num_epochs": {}}

# hyper-parameter candidates
epoch_candidates = [5, 10, 20, 30]
patience = 3  # early-stopping patience

for max_epochs in epoch_candidates:
    print(f"\n=== Training with max_epochs = {max_epochs} ===")
    run_key = f"epochs_{max_epochs}"
    experiment_data["num_epochs"][run_key] = {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "best_epoch": None,
    }

    # fresh model / optimiser / criterion
    model = GlyphModel(len(shapes), len(colors), n_clusters, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0

    # training loop with early stopping
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader_full:
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
        train_loss = running_loss / len(train_loader_full.dataset)
        val_loss, val_metrics, _, _ = evaluate(model, dev_loader_full, criterion)

        # record
        experiment_data["num_epochs"][run_key]["losses"]["train"].append(train_loss)
        experiment_data["num_epochs"][run_key]["losses"]["val"].append(val_loss)
        experiment_data["num_epochs"][run_key]["metrics"]["train"].append({})
        experiment_data["num_epochs"][run_key]["metrics"]["val"].append(val_metrics)

        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | CWA={val_metrics['CWA']:.3f} "
            f"SWA={val_metrics['SWA']:.3f} GCWA={val_metrics['GCWA']:.3f}"
        )

        # early-stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            experiment_data["num_epochs"][run_key]["best_epoch"] = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # reload best weights & evaluate on test
    model.load_state_dict(best_state)
    test_loss, test_metrics, test_preds, test_tgts = evaluate(
        model, test_loader_full, criterion
    )
    experiment_data["num_epochs"][run_key]["losses"]["test"] = test_loss
    experiment_data["num_epochs"][run_key]["metrics"]["test"] = test_metrics
    experiment_data["num_epochs"][run_key]["predictions"] = test_preds
    experiment_data["num_epochs"][run_key]["ground_truth"] = test_tgts

    print(
        f"Test: loss={test_loss:.4f} | "
        f"CWA={test_metrics['CWA']:.3f} SWA={test_metrics['SWA']:.3f} "
        f"GCWA={test_metrics['GCWA']:.3f}"
    )

# ------------------------------------------------------------------------- #
# persist experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
