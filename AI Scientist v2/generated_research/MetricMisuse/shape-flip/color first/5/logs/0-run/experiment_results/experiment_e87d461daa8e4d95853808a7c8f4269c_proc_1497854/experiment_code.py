#!/usr/bin/env python
import os, pathlib, random, time, copy, numpy as np, torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn

# -------------------------------------------------------------------------
# basic setup -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------------
# load SPR or synthetic ---------------------------------------------------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import SPR

        DATA_PATH = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(DATA_PATH)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        raise IOError from e


def build_synthetic_dataset(n_train=600, n_val=150, n_test=150):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def make_seq():
        L = random.randint(4, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def mk(n):
        return [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]

    return mk(n_train), mk(n_val), mk(n_test)


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    dataset_name = "SPR_BENCH"
    print("Loaded real SPR_BENCH.")
except IOError:
    print("Using synthetic data (real dataset not found).")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()
    dataset_name = "synthetic"

# -------------------------------------------------------------------------
# vocab & label maps ------------------------------------------------------
token2idx = {"<PAD>": 0}
for split in train_rows + dev_rows + test_rows:
    for tok in split["sequence"].split():
        if tok not in token2idx:
            token2idx[tok] = len(token2idx)

label2idx = {}
for r in train_rows + dev_rows + test_rows:
    if r["label"] not in label2idx:
        label2idx[r["label"]] = len(label2idx)

num_classes = len(label2idx)
print(f"Vocab={len(token2idx)}, Labels={num_classes}")


# -------------------------------------------------------------------------
# metrics -----------------------------------------------------------------
def count_color_variety(seq):
    return len({t[1] for t in seq.split()})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split()})


def weighted_acc(seqs, y_true, y_pred, weight_fn):
    w = [weight_fn(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc(seqs, y_true, y_pred, count_color_variety)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc(seqs, y_true, y_pred, count_shape_variety)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc(
        seqs, y_true, y_pred, lambda s: count_color_variety(s) + count_shape_variety(s)
    )


# -------------------------------------------------------------------------
# graph builders ----------------------------------------------------------
def seq_to_graph(seq: str, label: str, directed: bool = True) -> Data:
    toks = seq.split()
    n = len(toks)
    shapes = [t[0] for t in toks]
    colors = [t[1] for t in toks]
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    src, dst, etype = [], [], []

    def add_edge(i, j, rel):
        src.append(i)
        dst.append(j)
        etype.append(rel)

    # relation 0: neighbour in sequence
    for i in range(n - 1):
        add_edge(i, i + 1, 0)
        if directed:
            add_edge(i + 1, i, 0)

    # relation 1: same shape
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i] == shapes[j]:
                add_edge(i, j, 1)
                if directed:
                    add_edge(j, i, 1)

    # relation 2: same color
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                add_edge(i, j, 2)
                if directed:
                    add_edge(j, i, 2)

    if not src:  # rare fallback
        add_edge(0, 0, 0)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    return Data(
        x=node_feats,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label2idx[label]], dtype=torch.long),
        seq=seq,
    )


# -------------------------------------------------------------------------
# dataloaders builder -----------------------------------------------------
def build_dataloaders(directed: bool, batch_size: int = 128):
    train_graphs = [
        seq_to_graph(r["sequence"], r["label"], directed) for r in train_rows
    ]
    val_graphs = [seq_to_graph(r["sequence"], r["label"], directed) for r in dev_rows]
    test_graphs = [seq_to_graph(r["sequence"], r["label"], directed) for r in test_rows]
    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=batch_size),
        DataLoader(test_graphs, batch_size=batch_size),
    )


# -------------------------------------------------------------------------
# model -------------------------------------------------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, vocab, emb=64, hid=64, rels=3, classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv1 = RGCNConv(emb, hid, num_relations=rels)
        self.conv2 = RGCNConv(hid, hid, num_relations=rels)
        self.lin = nn.Linear(hid, classes)

    def forward(self, x, edge_index, edge_type, batch):
        h = torch.relu(self.conv1(self.embed(x), edge_index, edge_type))
        h = torch.relu(self.conv2(h, edge_index, edge_type))
        g = global_mean_pool(h, batch)
        return self.lin(g)


# -------------------------------------------------------------------------
# training utilities ------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    tot_loss, seqs, yt, yp = 0.0, [], [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
            preds = out.argmax(-1).cpu().tolist()
            yp.extend(preds)
            yt.extend(batch.y.squeeze().cpu().tolist())
            seqs.extend(batch.seq)
    n = len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, yt, yp)
    swa = shape_weighted_accuracy(seqs, yt, yp)
    cpx = complexity_weighted_accuracy(seqs, yt, yp)
    return tot_loss / n, cwa, swa, cpx, yp, yt, seqs


# -------------------------------------------------------------------------
# experiment dict ---------------------------------------------------------
experiment_data = {
    "edge_direction": {
        "directed": {
            "metrics": {
                "train": {"CWA": [], "SWA": [], "CmpWA": []},
                "val": {"CWA": [], "SWA": [], "CmpWA": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "undirected": {
            "metrics": {
                "train": {"CWA": [], "SWA": [], "CmpWA": []},
                "val": {"CWA": [], "SWA": [], "CmpWA": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# -------------------------------------------------------------------------
# training loop per variant ----------------------------------------------
def train_variant(name: str, directed: bool):
    print(f"\n--- Training variant: {name} (directed={directed}) ---")
    train_loader, val_loader, test_loader = build_dataloaders(directed)
    model = SPR_RGCN(len(token2idx), classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, best_state, wait, patience = float("inf"), None, 0, 7
    for epoch in range(1, 41):
        tr_loss, tr_cwa, tr_swa, tr_cpx, *_ = run_epoch(
            model, train_loader, criterion, opt
        )
        vl_loss, vl_cwa, vl_swa, vl_cpx, *_ = run_epoch(model, val_loader, criterion)

        exp_branch = experiment_data["edge_direction"][name]
        exp_branch["losses"]["train"].append(tr_loss)
        exp_branch["losses"]["val"].append(vl_loss)
        for k, v in zip(["CWA", "SWA", "CmpWA"], [tr_cwa, tr_swa, tr_cpx]):
            exp_branch["metrics"]["train"][k].append(v)
        for k, v in zip(["CWA", "SWA", "CmpWA"], [vl_cwa, vl_swa, vl_cpx]):
            exp_branch["metrics"]["val"][k].append(v)

        print(
            f"Epoch {epoch:02d} | tr_loss {tr_loss:.3f} vl_loss {vl_loss:.3f} vl_CmpWA {vl_cpx:.3f}"
        )
        if vl_loss < best_val - 1e-4:
            best_val, best_state, wait = vl_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break
    # load best, evaluate test
    if best_state is not None:
        model.load_state_dict(best_state)
    ts_loss, ts_cwa, ts_swa, ts_cpx, ts_pred, ts_true, _ = run_epoch(
        model, test_loader, criterion
    )
    exp_branch["predictions"] = ts_pred
    exp_branch["ground_truth"] = ts_true
    exp_branch["test_metrics"] = {
        "loss": ts_loss,
        "CWA": ts_cwa,
        "SWA": ts_swa,
        "CmpWA": ts_cpx,
    }
    print(
        f"TEST ({name}): loss={ts_loss:.3f} CWA={ts_cwa:.3f} SWA={ts_swa:.3f} CmpWA={ts_cpx:.3f}"
    )
    del model
    torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# run both variants -------------------------------------------------------
train_variant("directed", directed=True)
train_variant("undirected", directed=False)

# -------------------------------------------------------------------------
# save experiment data ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all metrics to working/experiment_data.npy")
