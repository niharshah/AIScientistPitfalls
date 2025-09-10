import os, random, string, time, pathlib, numpy as np, torch
from typing import List
from datasets import Dataset, DatasetDict
from torch import nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool

# ---------------------------------------------------------------------
# EXPERIMENT DATA STORAGE ---------------------------------------------
experiment_data = {
    "single_layer_gnn": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------------------------------------------------------------------
# WORK DIR & DEVICE ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------
# METRICS --------------------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def pcwa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    num = sum(w for w, yt, yp in zip(weights, y_true, y_pred) if yt == yp)
    den = sum(weights)
    return num / den if den else 0.0


# ---------------------------------------------------------------------
# DATA (load SPR benchmark if available, else synthetic) --------------
ROOT = pathlib.Path("./SPR_BENCH")
SHAPES = list(string.ascii_uppercase[:6])  # A-F
COLORS = list(map(str, range(1, 7)))  # 1-6


def gen_seq(max_len: int = 12) -> str:
    return " ".join(
        random.choice(SHAPES) + random.choice(COLORS)
        for _ in range(random.randint(4, max_len))
    )


def label_rule(seq: str) -> int:
    return int(count_shape_variety(seq) >= count_color_variety(seq))


def synthesize_split(n: int) -> Dataset:
    return Dataset.from_dict(
        {
            "sequence": [gen_seq() for _ in range(n)],
            "label": [label_rule(gen_seq()) for _ in range(n)],
        }
    )


def load_dataset_dict() -> DatasetDict:
    if ROOT.exists():
        from SPR import load_spr_bench

        return load_spr_bench(ROOT)
    print("SPR_BENCH not found – generating synthetic data.")
    return DatasetDict(
        {
            "train": synthesize_split(2000),
            "dev": synthesize_split(400),
            "test": synthesize_split(400),
        }
    )


dsets = load_dataset_dict()
num_classes = len(set(dsets["train"]["label"]))
print({k: len(v) for k, v in dsets.items()})

# ---------------------------------------------------------------------
# GRAPH CONVERSION -----------------------------------------------------
shape_to_id = {s: i for i, s in enumerate(SHAPES)}
color_to_id = {c: i for i, c in enumerate(COLORS)}
feat_dim = len(SHAPES) + len(COLORS) + 1  # shape one-hot + color one-hot + position


def seq_to_graph(seq: str) -> Data:
    toks = seq.strip().split()
    n = len(toks)
    x = torch.zeros((n, feat_dim), dtype=torch.float)
    for i, tok in enumerate(toks):
        x[i, shape_to_id[tok[0]]] = 1.0
        x[i, len(SHAPES) + color_to_id[tok[1]]] = 1.0
        x[i, -1] = i / (n - 1) if n > 1 else 0.0
    edges = (
        [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
        if n > 1
        else []
    )
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index, seq=seq)


class SPRGraphDataset(InMemoryDataset):
    def __init__(self, hf_dataset: Dataset):
        super().__init__(None, None, None)
        data_list = []
        for seq, label in zip(hf_dataset["sequence"], hf_dataset["label"]):
            g = seq_to_graph(seq)
            g.y = torch.tensor([label], dtype=torch.long)
            data_list.append(g)
        self.data, self.slices = self.collate(data_list)


# ---------------------------------------------------------------------
# DATA LOADERS ---------------------------------------------------------
train_data = SPRGraphDataset(dsets["train"])
val_data = SPRGraphDataset(dsets["dev"])
test_data = SPRGraphDataset(dsets["test"])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)


# ---------------------------------------------------------------------
# MODEL (Single-layer GNN) --------------------------------------------
class OneLayerGNN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv = GraphConv(in_dim, hid)
        self.lin = nn.Linear(hid, num_classes)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = OneLayerGNN(feat_dim, hid=64, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------
# TRAIN / EVAL ---------------------------------------------------------
def run_epoch(loader, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_seq, y_true, y_pred = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(1)
        correct += int((pred == data.y.view(-1)).sum())
        total += data.num_graphs
        all_seq.extend(data.seq)
        y_true.extend(data.y.view(-1).tolist())
        y_pred.extend(pred.tolist())
    return (
        total_loss / total,
        correct / total,
        pcwa(all_seq, y_true, y_pred),
    )


# ---------------------------------------------------------------------
EPOCHS = 5
start = time.time()
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_pc = run_epoch(train_loader, True)
    val_loss, val_acc, val_pc = run_epoch(val_loader, False)

    experiment_data["single_layer_gnn"]["SPR"]["losses"]["train"].append(
        (epoch, tr_loss)
    )
    experiment_data["single_layer_gnn"]["SPR"]["losses"]["val"].append(
        (epoch, val_loss)
    )
    experiment_data["single_layer_gnn"]["SPR"]["metrics"]["train"].append(
        (epoch, tr_pc)
    )
    experiment_data["single_layer_gnn"]["SPR"]["metrics"]["val"].append((epoch, val_pc))

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"val_PCWA={val_pc:.4f}"
    )


# ---------------------------------------------------------------------
# FINAL TEST -----------------------------------------------------------
def evaluate(loader):
    model.eval()
    all_seq, y_true, y_pred = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.argmax(1)
        all_seq.extend(data.seq)
        y_true.extend(data.y.view(-1).tolist())
        y_pred.extend(pred.tolist())
    metrics = {
        "PCWA": pcwa(all_seq, y_true, y_pred),
        "ACC": sum(int(y == p) for y, p in zip(y_true, y_pred)) / len(y_true),
    }
    return metrics, y_true, y_pred


test_metrics, y_t, y_p = evaluate(test_loader)
print("TEST METRICS:", test_metrics)

experiment_data["single_layer_gnn"]["SPR"]["predictions"] = y_p
experiment_data["single_layer_gnn"]["SPR"]["ground_truth"] = y_t

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Finished in {time.time()-start:.1f}s; results saved to {working_dir}")
