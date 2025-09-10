# Multi-Synthetic Dataset Generalisation Ablation – single-file script
import os, random, copy, numpy as np, torch, pathlib
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------------------------------------------------------
# Synthetic corpus builder ------------------------------------------------
def build_synthetic_dataset(n_items: int = 900) -> List[dict]:
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def make_seq():
        L = random.randint(4, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    rows = [
        {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
        for i in range(n_items)
    ]
    return rows


# Fix seeds and create datasets A,B,C ------------------------------------
dataset_A = build_synthetic_dataset()
random.seed(1)
dataset_B = build_synthetic_dataset()
random.seed(2)
dataset_C = build_synthetic_dataset()


# -------------------------------------------------------------------------
# Vocabulary and label mapping over union of all three datasets ----------
def all_tokens(rows):
    for r in rows:
        for tok in r["sequence"].split():
            yield tok


token2idx = {"<PAD>": 0}
for tok in all_tokens(dataset_A + dataset_B + dataset_C):
    token2idx.setdefault(tok, len(token2idx))
label2idx = {}
for r in dataset_A + dataset_B + dataset_C:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print(f"Vocab={len(token2idx)}, Labels={num_classes}")


# -------------------------------------------------------------------------
# Graph construction ------------------------------------------------------
def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    shapes = [t[0] for t in toks]
    colors = [t[1] for t in toks]
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    src, dst, etype = [], [], []
    # relation 0 – order
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [0, 0]
    # relation 1 – same shape
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i] == shapes[j]:
                src += [i, j]
                dst += [j, i]
                etype += [1, 1]
    # relation 2 – same colour
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                src += [i, j]
                dst += [j, i]
                etype += [2, 2]
    if not src:
        src, dst, etype = [0], [0], [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(
        x=node_feats,
        edge_index=edge_index,
        edge_type=torch.tensor(etype, dtype=torch.long),
        y=torch.tensor([label2idx[label]], dtype=torch.long),
        seq=seq,
    )


# -------------------------------------------------------------------------
# Weighted-accuracy helpers ----------------------------------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split()})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def weighted_acc(seqs, y_true, y_pred, w_fn):
    w = [w_fn(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------------------------------------------------------------------
# Data loaders ------------------------------------------------------------
def rows_to_loader(rows, batch_size=128, shuffle=False):
    graphs = [seq_to_graph(r["sequence"], r["label"]) for r in rows]
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


# -------------------------------------------------------------------------
# R-GCN Model -------------------------------------------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, vocab, embed=64, hid=64, rel=3, cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, padding_idx=0)
        self.c1 = RGCNConv(embed, hid, num_relations=rel)
        self.c2 = RGCNConv(hid, hid, num_relations=rel)
        self.lin = nn.Linear(hid, cls)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.emb(x)
        x = torch.relu(self.c1(x, edge_index, edge_type))
        x = torch.relu(self.c2(x, edge_index, edge_type))
        return self.lin(global_mean_pool(x, batch))


# -------------------------------------------------------------------------
# One epoch ---------------------------------------------------------------
def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total, seqs, y_true, y_pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        y_pred += preds
        y_true += batch.y.squeeze().cpu().tolist()
        seqs += batch.seq
    avg = total / len(loader.dataset)
    cwa = weighted_acc(seqs, y_true, y_pred, count_color_variety)
    swa = weighted_acc(seqs, y_true, y_pred, count_shape_variety)
    cmp = weighted_acc(
        seqs, y_true, y_pred, lambda s: count_color_variety(s) + count_shape_variety(s)
    )
    return avg, cwa, swa, cmp, y_pred, y_true, seqs


# -------------------------------------------------------------------------
# Training routine --------------------------------------------------------
def train_model(train_rows, val_rows, test_rows, max_epochs=40, patience=7):
    loaders = {
        "train": rows_to_loader(train_rows, shuffle=True),
        "val": rows_to_loader(val_rows),
        "test": rows_to_loader(test_rows),
    }
    model = SPR_RGCN(len(token2idx), cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, wait, best_state = float("inf"), 0, None
    logs = {"losses": {"train": [], "val": []}, "metrics": {"train": [], "val": []}}
    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, loaders["train"], criterion, opt)
        vl = run_epoch(model, loaders["val"], criterion)
        logs["losses"]["train"].append(tr[0])
        logs["losses"]["val"].append(vl[0])
        logs["metrics"]["train"].append(tr[3])
        logs["metrics"]["val"].append(vl[3])
        # early stop
        if vl[0] < best - 1e-4:
            best, best_state, wait = vl[0], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    ts = run_epoch(model, loaders["test"], criterion)
    return logs, ts


# -------------------------------------------------------------------------
# Run Ablation experiments ------------------------------------------------
experiment_data = {"MultiSyntheticGeneralization": {}}

# 1) Train A / Val B / Test C --------------------------------------------
print("\n=== Experiment: Train A | Val B | Test C ===")
logs1, test1 = train_model(dataset_A, dataset_B, dataset_C)
experiment_data["MultiSyntheticGeneralization"]["A_train_B_val_C_test"] = {
    "losses": logs1["losses"],
    "CmpWA_train": logs1["metrics"]["train"],
    "CmpWA_val": logs1["metrics"]["val"],
    "predictions": test1[4],
    "ground_truth": test1[5],
    "test_metrics": {
        "loss": test1[0],
        "CWA": test1[1],
        "SWA": test1[2],
        "CmpWA": test1[3],
    },
}

# 2) Train (A+B) / Val B / Test C ----------------------------------------
print("\n=== Experiment: Train A+B | Val B | Test C ===")
logs2, test2 = train_model(dataset_A + dataset_B, dataset_B, dataset_C)
experiment_data["MultiSyntheticGeneralization"]["AB_train_C_test"] = {
    "losses": logs2["losses"],
    "CmpWA_train": logs2["metrics"]["train"],
    "CmpWA_val": logs2["metrics"]["val"],
    "predictions": test2[4],
    "ground_truth": test2[5],
    "test_metrics": {
        "loss": test2[0],
        "CWA": test2[1],
        "SWA": test2[2],
        "CmpWA": test2[3],
    },
}

# -------------------------------------------------------------------------
# Save --------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
