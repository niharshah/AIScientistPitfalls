import os, random, string, pathlib, time, numpy as np, torch, warnings
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import List, Tuple
from datasets import load_dataset, DatasetDict

# -------------------- utility / reproducibility --------------------
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------- dir / device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# -------------------- metrics --------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1e-6, sum(weights))


# -------------------- dataset loading --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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


def generate_synthetic_dataset(n: int) -> Tuple[List[str], List[int]]:
    shapes = list(string.ascii_uppercase[:5])
    colors = list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(random.randint(0, 2))
    return seqs, labels


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    spr = load_spr_bench(data_root)
    print("Loaded real SPR_BENCH data")
except Exception as e:
    print("Falling back to synthetic data:", e)
    tr_s, tr_y = generate_synthetic_dataset(600)
    dv_s, dv_y = generate_synthetic_dataset(150)
    ts_s, ts_y = generate_synthetic_dataset(150)
    spr = DatasetDict(
        {
            "train": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
            "dev": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
            "test": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
        }
    )
    spr["train"] = spr["train"].add_column("sequence", tr_s).add_column("label", tr_y)
    spr["dev"] = spr["dev"].add_column("sequence", dv_s).add_column("label", dv_y)
    spr["test"] = spr["test"].add_column("sequence", ts_s).add_column("label", ts_y)


# -------------------- vocab --------------------
def build_vocabs(dataset) -> Tuple[dict, dict, dict]:
    shapes, colors, labels = set(), set(), set()
    for ex in dataset:
        for tok in ex["sequence"].split():
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
        {l: i for i, l in enumerate(sorted(labels))},
    )


shape2idx, color2idx, label2idx = build_vocabs(spr["train"])
num_shapes, len_colors, len_labels = len(shape2idx), len(color2idx), len(label2idx)


# -------------------- graph conversion --------------------
def sequence_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shape_idx = [shape2idx[t[0]] for t in toks]
    color_idx = [color2idx[t[1:]] for t in toks]
    x = torch.tensor(list(zip(shape_idx, color_idx)), dtype=torch.long)
    src, dst = [], []
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


train_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]
train_loader_global = DataLoader(train_graphs, batch_size=32, shuffle=True)
dev_loader_global = DataLoader(dev_graphs, batch_size=64)


# -------------------- model --------------------
class SPRGraphNet(nn.Module):
    def __init__(self, num_shapes, num_colors, num_classes, emb_dim=16, hidden=32):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, emb_dim)
        self.color_emb = nn.Embedding(num_colors, emb_dim)
        self.gnn1 = SAGEConv(emb_dim * 2, hidden)
        self.gnn2 = SAGEConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, data):
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        h = torch.cat([shp, col], dim=-1)
        h = self.gnn1(h, data.edge_index).relu()
        h = self.gnn2(h, data.edge_index).relu()
        hg = global_mean_pool(h, data.batch)
        return self.classifier(hg)


# -------------------- hyperparameter sweep --------------------
embed_dims = [8, 16, 32, 64]
epochs = 5
experiment_data = {"embedding_dim": {}}

for emb in embed_dims:
    print(f"\n=== Training with embedding_dim={emb} ===")
    model = SPRGraphNet(num_shapes, len_colors, len_labels, emb_dim=emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    metrics = {"train": [], "val": []}
    losses = {"train": [], "val": []}
    best_cwa2 = 0.0
    # epoch loop
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader_global:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_loss / len(train_loader_global.dataset)
        # validation
        model.eval()
        val_loss = 0
        ys = []
        preds = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader_global:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds.extend(out.argmax(-1).cpu().tolist())
                ys.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq)
        avg_val_loss = val_loss / len(dev_loader_global.dataset)
        cwa2 = complexity_weighted_accuracy(seqs, ys, preds)
        losses["train"].append(avg_train_loss)
        losses["val"].append(avg_val_loss)
        metrics["train"].append(0.0)  # skipped to save time
        metrics["val"].append(cwa2)
        print(
            f"Epoch {ep}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, CWA2={cwa2:.4f}"
        )
        if cwa2 > best_cwa2:
            best_cwa2 = cwa2
    # store
    experiment_data["embedding_dim"][f"emb_{emb}"] = {
        "metrics": metrics,
        "losses": losses,
        "predictions": preds,
        "ground_truth": ys,
    }

# -------------------- save --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
