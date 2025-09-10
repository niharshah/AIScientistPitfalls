# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, string, pathlib, numpy as np, torch, time
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from datasets import load_dataset, DatasetDict
from typing import List, Tuple

# --------------- reproducibility & device -----------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------- helpers ----------------------------------
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


# --------------- dataset loading --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def generate_synth(n: int) -> Tuple[List[str], List[int]]:
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
    print("Loaded real SPR_BENCH")
except Exception as e:
    print("Falling back to synthetic data:", e)
    tr_seq, tr_y = generate_synth(500)
    dv_seq, dv_y = generate_synth(120)
    ts_seq, ts_y = generate_synth(120)
    empty_ds = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    spr = DatasetDict(
        {
            "train": empty_ds.add_column("sequence", tr_seq).add_column("label", tr_y),
            "dev": empty_ds.add_column("sequence", dv_seq).add_column("label", dv_y),
            "test": empty_ds.add_column("sequence", ts_seq).add_column("label", ts_y),
        }
    )


# --------------- vocab creation ---------------------------
def build_vocabs(dataset):
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


# --------------- graph conversion -------------------------
def seq_to_graph(seq: str, label: int) -> Data:
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


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]


# --------------- model ------------------------------------
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


# --------------- training routine -------------------------
train_loader_global = DataLoader(train_graphs, batch_size=32, shuffle=True)
dev_loader_global = DataLoader(dev_graphs, batch_size=64)


def train_for_epochs(num_epochs: int) -> dict:
    model = SPRGraphNet(num_shapes, len_colors, len_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses_train, losses_val, metrics_val = [], [], []
    for epoch in range(1, num_epochs + 1):
        model.train()
        tot_loss = 0
        for batch in train_loader_global:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
        losses_train.append(tot_loss / len(train_loader_global.dataset))
        # validation
        model.eval()
        vloss, ys, preds, seqs = 0, [], [], []
        with torch.no_grad():
            for batch in dev_loader_global:
                batch = batch.to(device)
                out = model(batch)
                vloss += criterion(out, batch.y).item() * batch.num_graphs
                pred = out.argmax(dim=-1).cpu().tolist()
                ys.extend(batch.y.cpu().tolist())
                preds.extend(pred)
                seqs.extend(batch.seq)
        losses_val.append(vloss / len(dev_loader_global.dataset))
        metrics_val.append(complexity_weighted_accuracy(seqs, ys, preds))
        print(
            f"[{num_epochs}ep model] epoch {epoch}/{num_epochs}: val_loss={losses_val[-1]:.4f} CWA2={metrics_val[-1]:.4f}"
        )
    # final evaluation data
    return {
        "losses": {"train": losses_train, "val": losses_val},
        "metrics": {"val_cwa2": metrics_val},
        "predictions": preds,
        "ground_truth": ys,
    }


# --------------- hyperparameter tuning over epochs --------
epoch_options = [5, 15, 30, 50]
experiment_data = {"epochs": {"SPR_BENCH": {}}}
start = time.time()
for ep in epoch_options:
    experiment_data["epochs"]["SPR_BENCH"][str(ep)] = train_for_epochs(ep)
print("Total tuning time:", time.time() - start, "seconds")

# --------------- save -------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy to", working_dir)
