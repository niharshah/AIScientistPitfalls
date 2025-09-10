# multi_dataset_generalization.py
import os, random, string, time, pathlib, numpy as np, torch
from typing import List, Optional
from datasets import Dataset
from torch import nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool

# ---------------------------------------------------------------------
# EXPERIMENT STORAGE ---------------------------------------------------
experiment_data = {"multi_dataset_generalization": {}}

# ---------------------------------------------------------------------
# DEVICE ---------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# GLOBAL CONSTANTS -----------------------------------------------------
SHAPES = list(string.ascii_uppercase[:6])  # A-F
COLORS = list(map(str, range(1, 7)))  # 1-6
shape_to_id = {s: i for i, s in enumerate(SHAPES)}
color_to_id = {c: i for i, c in enumerate(COLORS)}
feat_dim = len(SHAPES) + len(COLORS) + 1  # shape one-hot + color one-hot + position


# ---------------------------------------------------------------------
# METRICS --------------------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split()))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split()))


def pcwa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(corr) / sum(weights) if sum(weights) else 0.0


# ---------------------------------------------------------------------
# SYNTHETIC DATA GENERATION -------------------------------------------
def label_rule(seq: str) -> int:
    return int(count_shape_variety(seq) >= count_color_variety(seq))


def synthesize_split(
    n_samples: int,
    max_len: int,
    rng: random.Random,
    shape_w: Optional[List[float]] = None,
    color_w: Optional[List[float]] = None,
) -> Dataset:
    s_w = shape_w if shape_w is not None else [1] * len(SHAPES)
    c_w = color_w if color_w is not None else [1] * len(COLORS)
    seqs, labels = [], []
    for _ in range(n_samples):
        length = rng.randint(4, max_len)
        toks = [
            rng.choices(SHAPES, weights=s_w)[0] + rng.choices(COLORS, weights=c_w)[0]
            for _ in range(length)
        ]
        seq = " ".join(toks)
        seqs.append(seq)
        labels.append(label_rule(seq))
    return Dataset.from_dict({"sequence": seqs, "label": labels})


def build_dataset_dict(seed: int, max_len: int, shape_w=None, color_w=None):
    rng = random.Random(seed)
    return {
        "train": synthesize_split(2000, max_len, rng, shape_w, color_w),
        "dev": synthesize_split(400, max_len, rng, shape_w, color_w),
        "test": synthesize_split(400, max_len, rng, shape_w, color_w),
    }


# ---------------------------------------------------------------------
# GRAPH CONVERSION -----------------------------------------------------
def seq_to_graph(seq: str) -> Data:
    tokens = seq.split()
    n = len(tokens)
    x = torch.zeros((n, feat_dim), dtype=torch.float)
    for i, tok in enumerate(tokens):
        s, c = tok[0], tok[1]
        x[i, shape_to_id[s]] = 1.0
        x[i, len(SHAPES) + color_to_id[c]] = 1.0
        x[i, -1] = i / (n - 1) if n > 1 else 0.0
    if n > 1:
        edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
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
# MODEL ---------------------------------------------------------------
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


# ---------------------------------------------------------------------
# TRAIN / EVAL --------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    tot_loss, tot, corr = 0.0, 0, 0
    seqs, y_true, y_pred = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        corr += int((pred == data.y.view(-1)).sum())
        tot += data.num_graphs
        seqs.extend(data.seq)
        y_true.extend(data.y.view(-1).tolist())
        y_pred.extend(pred.tolist())
    return (tot_loss / tot, corr / tot, pcwa(seqs, y_true, y_pred))


def evaluate(model, loader):
    model.eval()
    seqs, y_true, y_pred = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.argmax(dim=1)
        seqs.extend(data.seq)
        y_true.extend(data.y.view(-1).tolist())
        y_pred.extend(pred.tolist())
    cwa_num = sum(
        count_color_variety(s) if y == p else 0 for s, y, p in zip(seqs, y_true, y_pred)
    )
    cwa_den = sum(count_color_variety(s) for s in seqs)
    swa_num = sum(
        count_shape_variety(s) if y == p else 0 for s, y, p in zip(seqs, y_true, y_pred)
    )
    swa_den = sum(count_shape_variety(s) for s in seqs)
    return (
        {
            "PCWA": pcwa(seqs, y_true, y_pred),
            "CWA": cwa_num / cwa_den if cwa_den else 0.0,
            "SWA": swa_num / swa_den if swa_den else 0.0,
            "ACC": sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true),
        },
        y_true,
        y_pred,
        seqs,
    )


# ---------------------------------------------------------------------
# DATASET CONFIGS ------------------------------------------------------
configs = [
    dict(name="A_max8_balanced", seed=1, max_len=8, shape_w=None, color_w=None),
    dict(
        name="B_max12_shapeBias",
        seed=2,
        max_len=12,
        shape_w=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
        color_w=None,
    ),
    dict(
        name="C_max20_colorBias",
        seed=3,
        max_len=20,
        shape_w=None,
        color_w=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
    ),
]

# ---------------------------------------------------------------------
# MAIN LOOP ------------------------------------------------------------
EPOCHS = 5
start = time.time()
for cfg in configs:
    print(f"\n=== Dataset {cfg['name']} ===")
    # Build datasets
    dsdict = build_dataset_dict(
        cfg["seed"], cfg["max_len"], cfg["shape_w"], cfg["color_w"]
    )
    train_ds = SPRGraphDataset(dsdict["train"])
    dev_ds = SPRGraphDataset(dsdict["dev"])
    test_ds = SPRGraphDataset(dsdict["test"])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    # Model / optimiser
    model = GNNClassifier(feat_dim, hid=64, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Storage
    experiment_data["multi_dataset_generalization"][cfg["name"]] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "test_metrics": {},
    }
    rec = experiment_data["multi_dataset_generalization"][cfg["name"]]

    # Train
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_pc = run_epoch(model, train_loader, criterion, optimizer)
        dv_loss, dv_acc, dv_pc = run_epoch(model, dev_loader, criterion)

        rec["losses"]["train"].append((epoch, tr_loss))
        rec["losses"]["val"].append((epoch, dv_loss))
        rec["metrics"]["train"].append((epoch, tr_pc))
        rec["metrics"]["val"].append((epoch, dv_pc))

        print(
            f"Ep{epoch:02d}: tr_loss={tr_loss:.3f} dv_loss={dv_loss:.3f} dv_PCWA={dv_pc:.3f}"
        )

    # Final test
    tst_metrics, y_t, y_p, seqs = evaluate(model, test_loader)
    rec["test_metrics"] = tst_metrics
    rec["predictions"] = y_p
    rec["ground_truth"] = y_t
    print("Test:", tst_metrics)

# ---------------------------------------------------------------------
# SAVE -----------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nDone in {time.time()-start:.1f}s. Data saved to {working_dir}")
