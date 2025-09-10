import os, pathlib, random, copy, time, numpy as np, torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool  # <<<— GCN
import torch.nn as nn

# -------------------------------------------------------------------------
# working dir / device ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------------------------------------------------
# try to load real SPR_BENCH ----------------------------------------------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import SPR

        DATA_PATH = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(DATA_PATH)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        raise IOError from e


# synthetic fallback ------------------------------------------------------
def build_synthetic_dataset(n_train=600, n_val=150, n_test=150):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def make_seq():
        L = random.randint(4, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def make_split(n):
        return [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]

    def tag(lst):
        for i, r in enumerate(lst):
            r["id"] = i
        return lst

    return tag(make_split(n_train)), tag(make_split(n_val)), tag(make_split(n_test))


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    dataset_name = "SPR_BENCH"
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    train_rows, dev_rows, test_rows = build_synthetic_dataset()
    dataset_name = "synthetic"
    print("Using synthetic dataset (real dataset not found).")


# -------------------------------------------------------------------------
# vocab / label maps ------------------------------------------------------
def all_tokens(rows):
    for r in rows:
        for tok in r["sequence"].split():
            yield tok


token2idx = {"<PAD>": 0}
for tok in all_tokens(train_rows + dev_rows + test_rows):
    token2idx.setdefault(tok, len(token2idx))

label2idx = {}
for r in train_rows + dev_rows + test_rows:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print(f"Vocab size = {len(token2idx)} | #labels = {num_classes}")


# -------------------------------------------------------------------------
# metrics helpers ---------------------------------------------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def weighted_acc(weights, y_true, y_pred):
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) if weights else 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc([count_color_variety(s) for s in seqs], y_true, y_pred)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc([count_shape_variety(s) for s in seqs], y_true, y_pred)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc(
        [count_color_variety(s) + count_shape_variety(s) for s in seqs], y_true, y_pred
    )


# -------------------------------------------------------------------------
# graph construction (all edges collapsed to ONE relation) ----------------
def seq_to_single_rel_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    shapes = [t[0] for t in toks]
    colors = [t[1] if len(t) > 1 else "_" for t in toks]
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)

    src, dst = [], []
    # original relation 0: next / previous
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    # original relation 1: same shape
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i] == shapes[j]:
                src.extend([i, j])
                dst.extend([j, i])
    # original relation 2: same color
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                src.extend([i, j])
                dst.extend([j, i])

    if len(src) == 0:  # self loop fallback
        src, dst = [0], [0]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.zeros(
        edge_index.size(1), dtype=torch.long
    )  # single relation id 0
    return Data(
        x=node_feats,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label2idx[label]], dtype=torch.long),
        seq=seq,
    )


train_graphs = [seq_to_single_rel_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_single_rel_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_single_rel_graph(r["sequence"], r["label"]) for r in test_rows]

batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# -------------------------------------------------------------------------
# Ablation model: Two-layer GCN -------------------------------------------
class SPR_GCN(nn.Module):
    def __init__(self, vocab, embed_dim=64, hidden_dim=64, num_cls=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_cls)

    def forward(self, x, edge_index, edge_type, batch):  # edge_type ignored
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.lin(g)


# -------------------------------------------------------------------------
# training / eval loop ----------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    tot_loss, seqs, y_true, y_pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        if train_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch.y.squeeze().cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = tot_loss / len(loader.dataset)
    return (
        avg_loss,
        color_weighted_accuracy(seqs, y_true, y_pred),
        shape_weighted_accuracy(seqs, y_true, y_pred),
        complexity_weighted_accuracy(seqs, y_true, y_pred),
        y_pred,
        y_true,
        seqs,
    )


def train_model(
    model, train_loader, val_loader, test_loader, max_epochs=40, patience=7, lr=1e-3
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "loss_tr": [],
        "loss_val": [],
        "CWA_tr": [],
        "CWA_val": [],
        "SWA_tr": [],
        "SWA_val": [],
        "CmpWA_tr": [],
        "CmpWA_val": [],
    }
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optim)
        va = run_epoch(model, val_loader, criterion)
        history["loss_tr"].append(tr[0])
        history["loss_val"].append(va[0])
        history["CWA_tr"].append(tr[1])
        history["CWA_val"].append(va[1])
        history["SWA_tr"].append(tr[2])
        history["SWA_val"].append(va[2])
        history["CmpWA_tr"].append(tr[3])
        history["CmpWA_val"].append(va[3])
        print(
            f"Epoch {epoch:02d}: train_loss={tr[0]:.4f} val_loss={va[0]:.4f} "
            f"val_CmpWA={va[3]:.4f}"
        )
        if va[0] < best_val - 1e-4:
            best_val, best_state, wait = va[0], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break
    # load best and test
    if best_state is not None:
        model.load_state_dict(best_state)
    te = run_epoch(model, test_loader, criterion)
    print(f"TEST: loss={te[0]:.4f} CWA={te[1]:.4f} SWA={te[2]:.4f} CmpWA={te[3]:.4f}")
    return history, te


# -------------------------------------------------------------------------
# run ablation ------------------------------------------------------------
model_gcn = SPR_GCN(len(token2idx), num_cls=num_classes)
hist, test_stats = train_model(model_gcn, train_loader, val_loader, test_loader)

# -------------------------------------------------------------------------
# save experiment data ----------------------------------------------------
experiment_data = {
    "SingleRelation_GCN": {
        dataset_name: {
            "metrics": {
                "train": {
                    "CWA": hist["CWA_tr"],
                    "SWA": hist["SWA_tr"],
                    "CmpWA": hist["CmpWA_tr"],
                },
                "val": {
                    "CWA": hist["CWA_val"],
                    "SWA": hist["SWA_val"],
                    "CmpWA": hist["CmpWA_val"],
                },
            },
            "losses": {"train": hist["loss_tr"], "val": hist["loss_val"]},
            "predictions": test_stats[4],
            "ground_truth": test_stats[5],
            "test_metrics": {
                "loss": test_stats[0],
                "CWA": test_stats[1],
                "SWA": test_stats[2],
                "CmpWA": test_stats[3],
            },
        }
    }
}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all metrics to:", os.path.join(working_dir, "experiment_data.npy"))
