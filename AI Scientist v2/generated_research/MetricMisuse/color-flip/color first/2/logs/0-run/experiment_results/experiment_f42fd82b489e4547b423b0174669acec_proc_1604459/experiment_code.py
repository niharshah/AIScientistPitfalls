import os, pathlib, random, json, warnings, time

warnings.filterwarnings("ignore")

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- basic imports ----------
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- reproducibility ----------
def set_seed(sd=0):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)


set_seed(0)


# ---------- metric helpers ----------
def count_color_variety(sequence: str) -> int:
    return len({tok[1] for tok in sequence.strip().split() if len(tok) > 1})


def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in sequence.strip().split() if tok})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------- data loading ----------
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
        return [
            {
                "id": i,
                "sequence": " ".join(
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(random.randint(3, 9))
                ),
                "label": random.randint(0, 3),
            }
            for i in range(n)
        ]

    d = DatasetDict()
    for sp, n in zip(["train", "dev", "test"], [2000, 400, 400]):
        tmp_json_path = working_dir + f"/tmp_{sp}.jsonl"
        with open(tmp_json_path, "w") as f:
            for rec in gen_split(n):
                f.write(json.dumps(rec) + "\n")
        d[sp] = load_dataset("json", data_files=tmp_json_path, split="train")
    return d


spr = try_load_spr(DATA_PATH) or build_synthetic()
num_classes = len(set(spr["train"]["label"]))

# ---------- token → id maps & clustering ----------
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
shapes = sorted({t[0] for t in all_tokens})
colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(shapes)}  # 0 reserved for pad
color2id = {c: i + 1 for i, c in enumerate(colors)}
token_set = sorted(set(all_tokens))
token_vecs = np.array([[shape2id[t[0]], color2id[t[1]]] for t in token_set], float)
n_clusters = min(max(4, len(token_vecs) // 3), 32)
tok2cluster = {
    tok: int(c) + 1
    for tok, c in zip(
        token_set, KMeans(n_clusters, random_state=0).fit(token_vecs).labels_
    )
}


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = spr[split]["sequence"]
        self.labels = spr[split]["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        parts = self.seqs[idx].split()
        return {
            "shape": [shape2id[p[0]] for p in parts],
            "color": [color2id[p[1]] for p in parts],
            "cluster": [tok2cluster[p] for p in parts],
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
        )  # (B,L,3*emb)
        masked_sum = (e * mask.unsqueeze(-1)).sum(1)  # (B,3*emb)
        lengths = mask.sum(1, keepdim=True).clamp(min=1e-6)  # (B,1)
        pooled = masked_sum / lengths  # (B,3*emb)
        return self.ff(pooled)


# ---------- evaluation ----------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, preds, tgts, seqs = 0.0, [], [], []
    for batch in loader:
        # move to device
        tensor_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(
            tensor_batch["shape"],
            tensor_batch["color"],
            tensor_batch["cluster"],
            tensor_batch["mask"],
        )
        loss = criterion(logits, tensor_batch["labels"])
        tot_loss += loss.item() * tensor_batch["labels"].size(0)
        pred = logits.argmax(1).cpu().tolist()
        preds.extend(pred)
        tgts.extend(tensor_batch["labels"].cpu().tolist())
        seqs.extend(batch["seqs"])
    avg_loss = tot_loss / len(loader.dataset)
    metrics = {
        "CWA": color_weighted_accuracy(seqs, tgts, preds),
        "SWA": shape_weighted_accuracy(seqs, tgts, preds),
        "GCWA": glyph_complexity_weighted_accuracy(seqs, tgts, preds),
    }
    return avg_loss, metrics, preds, tgts


# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": [], "test": []},
        "predictions": [],
        "ground_truth": [],
        "hyperparams": [],
    }
}

# ---------- hyperparameter sweep ----------
learning_rates = [3e-4, 1e-4, 3e-5]  # reduced for runtime
epochs = 3
best_idx, best_gcwa = -1, -1.0

for run_idx, lr in enumerate(learning_rates):
    print(f"\n===== training run {run_idx+1} | lr={lr:.1e} =====")
    set_seed(run_idx)
    model = GlyphModel(len(shapes), len(colors), n_clusters, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_metrics_hist = [], [], []
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            tensor_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(
                tensor_batch["shape"],
                tensor_batch["color"],
                tensor_batch["cluster"],
                tensor_batch["mask"],
            )
            loss = criterion(logits, tensor_batch["labels"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * tensor_batch["labels"].size(0)
        train_loss = ep_loss / len(train_loader.dataset)

        val_loss, val_m, _, _ = evaluate(model, dev_loader, criterion)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | CWA={val_m['CWA']:.3f} | "
            f"SWA={val_m['SWA']:.3f} | GCWA={val_m['GCWA']:.3f}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_hist.append(val_m)

    # choose best by dev GCWA after final epoch
    final_gcwa = val_metrics_hist[-1]["GCWA"]
    if final_gcwa > best_gcwa:
        best_gcwa, best_idx = final_gcwa, run_idx

    # evaluate on test set
    test_loss, test_m, test_preds, test_tgts = evaluate(model, test_loader, criterion)
    print(
        f"Test | loss={test_loss:.4f} | CWA={test_m['CWA']:.3f} | "
        f"SWA={test_m['SWA']:.3f} | GCWA={test_m['GCWA']:.3f}"
    )

    # store experiment data
    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train"].append([])  # train metrics per-batch omitted
    ed["metrics"]["val"].append(val_metrics_hist)
    ed["metrics"]["test"].append(test_m)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["val"].append(val_losses)
    ed["losses"]["test"].append(test_loss)
    ed["predictions"].append(test_preds)
    ed["ground_truth"].append(test_tgts)
    ed["hyperparams"].append({"learning_rate": lr, "epochs": epochs})

print(
    f"\nBest learning rate: {learning_rates[best_idx]:.1e} "
    f"with dev GCWA {best_gcwa:.3f}"
)

# ---------- persist ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)
