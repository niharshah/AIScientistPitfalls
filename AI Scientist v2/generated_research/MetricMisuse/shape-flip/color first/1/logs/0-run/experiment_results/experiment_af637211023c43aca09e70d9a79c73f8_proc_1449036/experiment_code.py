# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, string, time, warnings, gc
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# -------------------- mandatory working dir -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- device handling -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- metrics ----------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# -------------------- dataset loading -------------------------
def load_spr_bench() -> "datasets.DatasetDict|None":  # type: ignore
    try:
        from datasets import load_dataset, DatasetDict
    except ImportError:
        return None
    root = pathlib.Path("SPR_BENCH")
    if not root.exists():
        return None

    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    from datasets import DatasetDict

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def make_synthetic(nt=500, nd=150, nte=200):
    shapes = list(string.ascii_uppercase[:8])
    colors = list(string.ascii_lowercase[:8])

    def gen(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(4, 18)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seq = " ".join(toks)
            label = int(sum(t[0] == toks[0][0] for t in toks) > L / 2)
            seqs.append(seq)
            labels.append(label)
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(gen(nt)),
            "dev": Dataset.from_dict(gen(nd)),
            "test": Dataset.from_dict(gen(nte)),
        }
    )


dset = load_spr_bench() or make_synthetic()
print("Loaded dataset sizes:", {k: len(v) for k, v in dset.items()})

# ------------------- vocab construction -----------------------
shape2idx, color2idx = {}, {}


def add_tok(tok):
    if tok[0] not in shape2idx:
        shape2idx[tok[0]] = len(shape2idx)
    if tok[1] not in color2idx:
        color2idx[tok[1]] = len(color2idx)


for seq in dset["train"]["sequence"]:
    for tok in seq.split():
        add_tok(tok)
n_shapes, n_colors = len(shape2idx), len(color2idx)
num_classes = len(set(dset["train"]["label"]))
print(f"Vocab sizes -> shapes:{n_shapes} colors:{n_colors} classes:{num_classes}")

# ------------------- seq -> graph -----------------------------
REL_CONSEC, REL_SAME_SHAPE, REL_SAME_COLOR = 0, 1, 2


def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shp_ids = [shape2idx[t[0]] for t in toks]
    col_ids = [color2idx[t[1]] for t in toks]
    x = torch.tensor(list(zip(shp_ids, col_ids)), dtype=torch.long)

    src, dst, etype = [], [], []
    # consecutive edges (bidirectional)
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etype.extend([REL_CONSEC] * 2)
    # same shape
    by_shape = {}
    for i, s in enumerate(shp_ids):
        by_shape.setdefault(s, []).append(i)
    for ids in by_shape.values():
        for i in ids:
            for j in ids:
                if i < j:
                    src.extend([i, j])
                    dst.extend([j, i])
                    etype.extend([REL_SAME_SHAPE] * 2)
    # same color
    by_color = {}
    for i, c in enumerate(col_ids):
        by_color.setdefault(c, []).append(i)
    for ids in by_color.values():
        for i in ids:
            for j in ids:
                if i < j:
                    src.extend([i, j])
                    dst.extend([j, i])
                    etype.extend([REL_SAME_COLOR] * 2)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    data = Data(
        x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([label])
    )
    data.seq_raw = seq  # keep for metric calc
    return data


train_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dset["train"]["sequence"], dset["train"]["label"])
]
dev_graphs = [
    seq_to_graph(s, l) for s, l in zip(dset["dev"]["sequence"], dset["dev"]["label"])
]
test_graphs = [
    seq_to_graph(s, l) for s, l in zip(dset["test"]["sequence"], dset["test"]["label"])
]


# ---------------------- model ---------------------------------
class RGCNClassifier(nn.Module):
    def __init__(
        self,
        n_shapes: int,
        n_colors: int,
        emb: int = 32,
        hid: int = 64,
        n_cls: int = 2,
        num_rels: int = 3,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb)
        self.color_emb = nn.Embedding(n_colors, emb)
        self.lin_in = nn.Linear(emb * 2, hid)
        self.rgcn1 = RGCNConv(hid, hid, num_rels)
        self.rgcn2 = RGCNConv(hid, hid, num_rels)
        self.rgcn3 = RGCNConv(hid, hid, num_rels)
        self.out = nn.Linear(hid, n_cls)

    def forward(self, data):
        sh = self.shape_emb(data.x[:, 0])
        co = self.color_emb(data.x[:, 1])
        x = F.relu(self.lin_in(torch.cat([sh, co], dim=-1)))
        x = F.relu(self.rgcn1(x, data.edge_index, data.edge_type))
        x = F.relu(self.rgcn2(x, data.edge_index, data.edge_type))
        x = F.relu(self.rgcn3(x, data.edge_index, data.edge_type))
        x = global_mean_pool(x, data.batch)
        return self.out(x)


# ------------------ experiment log dict -----------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------ training / evaluation ---------------------
def evaluate(model, loader, crit):
    model.eval()
    tot_loss = 0
    preds = []
    gts = []
    seqs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = crit(logits, batch.y)
            tot_loss += loss.item() * batch.num_graphs
            p = logits.argmax(-1).cpu().tolist()
            g = batch.y.cpu().tolist()
            seqs.extend(batch.seq_raw)
            preds.extend(p)
            gts.extend(g)
    avg_loss = tot_loss / len(loader.dataset)
    acc = np.mean([p == g for p, g in zip(preds, gts)])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hpa = harmonic_poly_accuracy(cwa, swa)
    return avg_loss, acc, cwa, swa, hpa, preds, gts, seqs


def run_experiment(
    lr: float, weight_decay: float = 1e-4, max_epochs: int = 15, patience: int = 3
):
    model = RGCNClassifier(n_shapes, n_colors, n_cls=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=64)

    best_hpa = -1
    best_state = None
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = crit(logits, batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
        tr_loss = tot_loss / len(train_loader.dataset)

        dev_loss, acc, cwa, swa, hpa, *_ = evaluate(model, dev_loader, crit)
        print(
            f"LR {lr} Epoch {epoch}: validation_loss = {dev_loss:.4f} | acc={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}"
        )

        experiment_data["SPR"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR"]["losses"]["val"].append(dev_loss)
        experiment_data["SPR"]["metrics"]["val"].append(
            {"acc": acc, "CWA": cwa, "SWA": swa, "HPA": hpa}
        )
        experiment_data["SPR"]["epochs"].append(epoch)

        if hpa > best_hpa:
            best_hpa = hpa
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break
    model.load_state_dict(best_state)
    return model, best_hpa


best_model = None
best_dev_hpa = -1
for lr in [1e-3, 3e-3]:
    print(f"\n========= Running LR {lr} =========")
    model, hpa = run_experiment(lr)
    if hpa > best_dev_hpa:
        best_dev_hpa = hpa
        best_model = model

# ------------------ final test evaluation ---------------------
test_loader = DataLoader(test_graphs, batch_size=64)
crit = nn.CrossEntropyLoss()
test_loss, acc, cwa, swa, hpa, preds, gts, seqs = evaluate(
    best_model, test_loader, crit
)
print(
    f"\nTEST RESULTS -> loss:{test_loss:.4f} acc:{acc:.3f} CWA:{cwa:.3f} SWA:{swa:.3f} HPA:{hpa:.3f}"
)

experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
