# Edge–Removal Ablation: each token is an isolated node
import os, random, string, numpy as np, torch, pathlib, time
from typing import List
from datasets import Dataset, DatasetDict
from torch import nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool

# ---------------------------------------------------------------------
# EXPERIMENT REGISTRY & STORAGE
experiment_data = {
    "NoEdges": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# METRICS --------------------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def pcwa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ---------------------------------------------------------------------
# DATA (load SPR or synthesise) ---------------------------------------
ROOT = pathlib.Path("./SPR_BENCH")
SHAPES = list(string.ascii_uppercase[:6])  # A-F
COLORS = list(map(str, range(1, 7)))  # 1-6


def gen_seq(max_len: int = 12) -> str:
    n = random.randint(4, max_len)
    return " ".join(random.choice(SHAPES) + random.choice(COLORS) for _ in range(n))


def label_rule(seq: str) -> int:
    return int(count_shape_variety(seq) >= count_color_variety(seq))


def synthesize_split(n: int) -> Dataset:
    seqs, labels = [], []
    for _ in range(n):
        s = gen_seq()
        seqs.append(s)
        labels.append(label_rule(s))
    return Dataset.from_dict({"sequence": seqs, "label": labels})


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
# GRAPH CONVERSION (No edges) -----------------------------------------
shape_to_id = {s: i for i, s in enumerate(SHAPES)}
color_to_id = {c: i for i, c in enumerate(COLORS)}
feat_dim = len(SHAPES) + len(COLORS) + 1  # shape one-hot + color one-hot + pos


def seq_to_graph_no_edges(seq: str) -> Data:
    toks = seq.strip().split()
    n = len(toks)
    x = torch.zeros((n, feat_dim), dtype=torch.float)
    for i, tok in enumerate(toks):
        s, c = tok[0], tok[1]
        x[i, shape_to_id[s]] = 1.0
        x[i, len(SHAPES) + color_to_id[c]] = 1.0
        x[i, -1] = i / (n - 1) if n > 1 else 0.0
    edge_index = torch.empty((2, 0), dtype=torch.long)  # <-- NO EDGES
    return Data(x=x, edge_index=edge_index, seq=seq)


class SPRGraphDataset(InMemoryDataset):
    def __init__(self, hf_dataset: Dataset):
        super().__init__(None, None, None)
        data_list = []
        for s, y in zip(hf_dataset["sequence"], hf_dataset["label"]):
            g = seq_to_graph_no_edges(s)
            g.y = torch.tensor([y], dtype=torch.long)
            data_list.append(g)
        self.data, self.slices = self.collate(data_list)


# ---------------------------------------------------------------------
# LOADERS --------------------------------------------------------------
train_data = SPRGraphDataset(dsets["train"])
val_data = SPRGraphDataset(dsets["dev"])
test_data = SPRGraphDataset(dsets["test"])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)


# ---------------------------------------------------------------------
# MODEL ----------------------------------------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hid)
        self.conv2 = GraphConv(hid, hid)
        self.lin = nn.Linear(hid, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = GNNClassifier(feat_dim, hid=64, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------
# TRAIN / EVAL ---------------------------------------------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss, tot, correct = 0.0, 0, 0
    all_seq, y_true, y_pred = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * data.num_graphs
        pred = out.argmax(1)
        correct += int((pred == data.y.view(-1)).sum())
        tot += data.num_graphs
        all_seq.extend(data.seq)
        y_true.extend(data.y.view(-1).tolist())
        y_pred.extend(pred.tolist())
    return (
        tot_loss / tot,
        correct / tot,
        pcwa(all_seq, y_true, y_pred),
    )


EPOCHS = 5
start = time.time()
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_pc = run_epoch(train_loader, True)
    vl_loss, vl_acc, vl_pc = run_epoch(val_loader, False)
    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} val_PCWA={vl_pc:.4f}"
    )
    experiment_data["NoEdges"]["SPR"]["losses"]["train"].append((ep, tr_loss))
    experiment_data["NoEdges"]["SPR"]["losses"]["val"].append((ep, vl_loss))
    experiment_data["NoEdges"]["SPR"]["metrics"]["train"].append((ep, tr_pc))
    experiment_data["NoEdges"]["SPR"]["metrics"]["val"].append((ep, vl_pc))


# ---------------------------------------------------------------------
# TEST EVALUATION ------------------------------------------------------
def compute_metrics(loader):
    model.eval()
    seqs, ys, ps = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.argmax(1)
        seqs.extend(data.seq)
        ys.extend(data.y.view(-1).tolist())
        ps.extend(pred.tolist())
    cwa_num = sum(
        count_color_variety(s) if y == p else 0 for s, y, p in zip(seqs, ys, ps)
    )
    cwa_den = sum(count_color_variety(s) for s in seqs)
    swa_num = sum(
        count_shape_variety(s) if y == p else 0 for s, y, p in zip(seqs, ys, ps)
    )
    swa_den = sum(count_shape_variety(s) for s in seqs)
    metrics = {
        "PCWA": pcwa(seqs, ys, ps),
        "CWA": cwa_num / cwa_den if cwa_den else 0.0,
        "SWA": swa_num / swa_den if swa_den else 0.0,
        "ACC": sum(int(y == p) for y, p in zip(ys, ps)) / len(ys),
    }
    return metrics, ys, ps, seqs


test_metrics, y_true, y_pred, seqs = compute_metrics(test_loader)
print("TEST METRICS:", test_metrics)
experiment_data["NoEdges"]["SPR"]["predictions"] = y_pred
experiment_data["NoEdges"]["SPR"]["ground_truth"] = y_true

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Finished in {time.time()-start:.1f}s. Data saved to {working_dir}")
