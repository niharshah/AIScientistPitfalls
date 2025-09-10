# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, time, pathlib, random, string, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ───── basic folders ───────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ───── device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ───── experiment-data container ───────────────────────────────
experiment_data = {
    "SPR_noSeqEdge": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ───── helpers: metrics ────────────────────────────────────────
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    score = sum(w for w, yt, yp in zip(weights, y_true, y_pred) if yt == yp)
    return score / max(1e-9, sum(weights))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    score = sum(w for w, yt, yp in zip(weights, y_true, y_pred) if yt == yp)
    return score / max(1e-9, sum(weights))


def dual_weighted_accuracy(seqs, y_true, y_pred):
    return 0.5 * (
        color_weighted_accuracy(seqs, y_true, y_pred)
        + shape_weighted_accuracy(seqs, y_true, y_pred)
    )


# ───── data loading ────────────────────────────────────────────
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def make_synthetic(n):
    shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seqs.append(
            " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        )
        labels.append(random.randint(0, 2))
    return seqs, labels


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    spr = load_spr_bench(data_root)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic:", e)
    tr_s, tr_y = make_synthetic(1200)
    dv_s, dv_y = make_synthetic(300)
    ts_s, ts_y = make_synthetic(300)
    blank = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    spr = DatasetDict(
        {
            "train": blank.add_column("sequence", tr_s).add_column("label", tr_y),
            "dev": blank.add_column("sequence", dv_s).add_column("label", dv_y),
            "test": blank.add_column("sequence", ts_s).add_column("label", ts_y),
        }
    )


# ───── vocabularies ────────────────────────────────────────────
def build_vocab(dataset):
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


shape2i, color2i, label2i = build_vocab(spr["train"])
NUM_SH, NUM_CL, NUM_LB, MAX_POS = len(shape2i), len(color2i), len(label2i), 25


# ───── graph builder (ablation: no sequential edges) ───────────
def seq_to_graph(seq, label):
    tokens = seq.split()
    n = len(tokens)
    shp = [shape2i[t[0]] for t in tokens]
    clr = [color2i[t[1:]] for t in tokens]
    pos = list(range(n))

    src, dst, etype = [], [], []

    # same-colour edges (rel-type 1)
    colour_groups = {}
    for idx, tok in enumerate(tokens):
        colour_groups.setdefault(tok[1:], []).append(idx)
    for idxs in colour_groups.values():
        for i in idxs:
            for j in idxs:
                if i < j:
                    src.extend([i, j])
                    dst.extend([j, i])
                    etype.extend([1, 1])

    # same-shape edges (rel-type 2)
    shape_groups = {}
    for idx, tok in enumerate(tokens):
        shape_groups.setdefault(tok[0], []).append(idx)
    for idxs in shape_groups.values():
        for i in idxs:
            for j in idxs:
                if i < j:
                    src.extend([i, j])
                    dst.extend([j, i])
                    etype.extend([2, 2])

    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = (
        torch.tensor(etype, dtype=torch.long)
        if etype
        else torch.empty((0,), dtype=torch.long)
    )

    x = torch.tensor(list(zip(shp, clr, pos)), dtype=torch.long)
    y = torch.tensor([label2i[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["test"]]


# ───── model ───────────────────────────────────────────────────
class SPR_RGCN(nn.Module):
    def __init__(self, emb_dim=32, hid=128):
        super().__init__()
        self.shape_emb = nn.Embedding(NUM_SH, emb_dim)
        self.color_emb = nn.Embedding(NUM_CL, emb_dim)
        self.pos_emb = nn.Embedding(MAX_POS, emb_dim)
        in_dim = emb_dim * 3
        self.rg1 = RGCNConv(in_dim, hid, num_relations=3)
        self.rg2 = RGCNConv(hid, hid, num_relations=3)
        self.cls = nn.Linear(hid, NUM_LB)

    def forward(self, batch):
        h = torch.cat(
            [
                self.shape_emb(batch.x[:, 0]),
                self.color_emb(batch.x[:, 1]),
                self.pos_emb(batch.x[:, 2].clamp(max=MAX_POS - 1)),
            ],
            dim=-1,
        )
        h = self.rg1(h, batch.edge_index, batch.edge_type).relu()
        h = self.rg2(h, batch.edge_index, batch.edge_type).relu()
        hg = global_mean_pool(h, batch.batch)  # graph-level embedding
        return self.cls(hg)  # BUGFIX: classify graph embedding


# ───── training / evaluation loops ─────────────────────────────
def run_experiment(epochs=20, batch_size=64, lr=1e-3, patience=5):
    model = SPR_RGCN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tr_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_graphs, batch_size=batch_size)

    best_val, strikes = float("inf"), 0
    for epoch in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(tr_loader.dataset)
        experiment_data["SPR_noSeqEdge"]["losses"]["train"].append(total_loss)

        # ---- VALID ----
        model.eval()
        val_loss, ys, ps, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y.squeeze())
                val_loss += loss.item() * batch.num_graphs
                ps.extend(logits.argmax(1).cpu().tolist())
                ys.extend(batch.y.squeeze().cpu().tolist())
                seqs.extend(batch.seq)
        val_loss /= len(val_loader.dataset)
        experiment_data["SPR_noSeqEdge"]["losses"]["val"].append(val_loss)

        cwa = color_weighted_accuracy(seqs, ys, ps)
        swa = shape_weighted_accuracy(seqs, ys, ps)
        dwa = dual_weighted_accuracy(seqs, ys, ps)
        experiment_data["SPR_noSeqEdge"]["metrics"]["val"].append(
            {"epoch": epoch, "CWA": cwa, "SWA": swa, "DWA": dwa}
        )

        print(
            f"Epoch {epoch:3d}  train_loss={total_loss:.4f}  val_loss={val_loss:.4f}  "
            f"CWA={cwa:.3f}  SWA={swa:.3f}  DWA={dwa:.3f}"
        )

        # early-stop
        if val_loss < best_val:
            best_val, strikes = val_loss, 0
        else:
            strikes += 1
            if strikes >= patience:
                print("Early stopping.")
                break

    # ---- TEST ----
    test_loader = DataLoader(test_graphs, batch_size=128)
    model.eval()
    ys, ps, seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            ps.extend(logits.argmax(1).cpu().tolist())
            ys.extend(batch.y.squeeze().cpu().tolist())
            seqs.extend(batch.seq)
    cwa = color_weighted_accuracy(seqs, ys, ps)
    swa = shape_weighted_accuracy(seqs, ys, ps)
    dwa = dual_weighted_accuracy(seqs, ys, ps)
    experiment_data["SPR_noSeqEdge"]["metrics"]["test"] = {
        "CWA": cwa,
        "SWA": swa,
        "DWA": dwa,
    }
    experiment_data["SPR_noSeqEdge"]["predictions"] = ps
    experiment_data["SPR_noSeqEdge"]["ground_truth"] = ys
    print("TEST  →  CWA={:.3f}  SWA={:.3f}  DWA={:.3f}".format(cwa, swa, dwa))


# ───── run everything ──────────────────────────────────────────
start = time.time()
run_experiment()
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved all experiment data to", os.path.join(working_dir, "experiment_data.npy"))
print("Total elapsed:", round(time.time() - start, 2), "s")
