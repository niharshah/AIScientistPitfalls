import os, random, string, numpy as np, torch
from torch import nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.loader import DataLoader  # <-- bug-fix: correct loader
from datasets import Dataset, DatasetDict
import pathlib
from typing import List

# ---------------------------------------------------------------------
# GLOBAL SET-UP --------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# container for logging ------------------------------------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------------------------------------------------------------
# METRICS --------------------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def pcwa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def cwa(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def swa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ---------------------------------------------------------------------
# DATA -----------------------------------------------------------------
ROOT = pathlib.Path("./SPR_BENCH")
SHAPES = list(string.ascii_uppercase[:6])  # A-F
COLORS = list(map(str, range(1, 7)))  # 1-6


def gen_seq(max_len=12):
    ln = random.randint(4, max_len)
    return " ".join(random.choice(SHAPES) + random.choice(COLORS) for _ in range(ln))


def label_rule(seq: str) -> int:
    return int(count_shape_variety(seq) >= count_color_variety(seq))


def synthesize_split(n_samples: int) -> Dataset:
    seqs, labels = [], []
    for _ in range(n_samples):
        s = gen_seq()
        seqs.append(s)
        labels.append(label_rule(s))
    return Dataset.from_dict({"sequence": seqs, "label": labels})


def load_or_generate() -> DatasetDict:
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


dsets = load_or_generate()
num_classes = len(set(dsets["train"]["label"]))
print({k: len(v) for k, v in dsets.items()})

# ---------------------------------------------------------------------
# GRAPH CONVERSION -----------------------------------------------------
shape_to_id = {s: i for i, s in enumerate(SHAPES)}
color_to_id = {c: i for i, c in enumerate(COLORS)}
feat_dim = len(SHAPES) + len(COLORS) + 1  # one-hots + position scalar


def seq_to_graph(seq: str) -> Data:
    toks = seq.strip().split()
    n = len(toks)
    x = torch.zeros((n, feat_dim), dtype=torch.float)
    for i, tok in enumerate(toks):
        s, c = tok[0], tok[1]
        x[i, shape_to_id[s]] = 1.0
        x[i, len(SHAPES) + color_to_id[c]] = 1.0
        x[i, -1] = i / (n - 1) if n > 1 else 0.0
    # bi-directional chain edges
    edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = Data(x=x, edge_index=edge_index)
    data.seq = seq
    return data


class SPRGraphDataset(InMemoryDataset):
    def __init__(self, hf_ds: Dataset):
        super().__init__(None, None, None)
        graphs = []
        for seq, label in zip(hf_ds["sequence"], hf_ds["label"]):
            g = seq_to_graph(seq)
            g.y = torch.tensor([label], dtype=torch.long)
            graphs.append(g)
        self.data, self.slices = self.collate(graphs)
        self.seqs = hf_ds["sequence"]


# ---------------------------------------------------------------------
train_ds = SPRGraphDataset(dsets["train"])
val_ds = SPRGraphDataset(dsets["dev"])
test_ds = SPRGraphDataset(dsets["test"])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)


# ---------------------------------------------------------------------
# MODEL ----------------------------------------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden)
        self.conv2 = GraphConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = GNNClassifier(feat_dim, hidden=64, n_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------------------
# TRAIN / EVAL ---------------------------------------------------------
def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    tot_loss, tot, correct = 0.0, 0, 0
    seqs, ys, preds = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += int((pred == data.y.view(-1)).sum().item())
        tot += data.num_graphs
        seqs.extend(data.seq)
        ys.extend(data.y.view(-1).cpu().tolist())
        preds.extend(pred.cpu().tolist())
    avg_loss = tot_loss / tot
    acc = correct / tot
    return avg_loss, acc, pcwa(seqs, ys, preds)


EPOCHS = 5
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_pc = run_epoch(train_loader, True)
    val_loss, val_acc, val_pc = run_epoch(val_loader, False)
    experiment_data["SPR"]["losses"]["train"].append((ep, tr_loss))
    experiment_data["SPR"]["losses"]["val"].append((ep, val_loss))
    experiment_data["SPR"]["metrics"]["train"].append((ep, tr_pc))
    experiment_data["SPR"]["metrics"]["val"].append((ep, val_pc))
    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_PCWA={val_pc:.4f}"
    )


# ---------------------------------------------------------------------
# FINAL TEST -----------------------------------------------------------
def eval_loader(loader):
    model.eval()
    seqs, ys, preds = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.argmax(dim=1)
        seqs.extend(data.seq)
        ys.extend(data.y.view(-1).cpu().tolist())
        preds.extend(pred.cpu().tolist())
    return (
        {
            "PCWA": pcwa(seqs, ys, preds),
            "CWA": cwa(seqs, ys, preds),
            "SWA": swa(seqs, ys, preds),
        },
        ys,
        preds,
        seqs,
    )


test_metrics, y_true, y_pred, seqs = eval_loader(test_loader)
experiment_data["SPR"]["predictions"] = y_pred
experiment_data["SPR"]["ground_truth"] = y_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("TEST METRICS:", test_metrics)
