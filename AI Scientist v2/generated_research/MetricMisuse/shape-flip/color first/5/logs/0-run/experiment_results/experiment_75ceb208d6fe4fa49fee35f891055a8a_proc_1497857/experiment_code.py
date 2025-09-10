# constant_node_feature_ablation.py
import os, pathlib, random, copy, numpy as np, torch, time
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn

# -------------------------------------------------------------------------
# working dir & torch device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# load real SPR or fall back to synthetic ---------------------------------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import SPR

        DATA_PATH = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(DATA_PATH)
        return dset["train"], dset["dev"], dset["test"]
    except Exception:
        raise IOError


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
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    print("Using synthetic dataset (real dataset not found).")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# metrics helpers ---------------------------------------------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


# -------------------------------------------------------------------------
# graph construction (TOKEN-BLIND) ----------------------------------------
def seq_to_graph_tokenblind(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    shapes = [t[0] for t in toks]
    colors = [t[1] if len(t) > 1 else "_" for t in toks]

    # every node gets identical index 0 -> same learnable vector
    node_feats = torch.zeros(n, dtype=torch.long)  # all zeros

    src, dst, etype = [], [], []
    # relation 0: sequential neighbours
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etype.extend([0, 0])
    # relation 1: same shape
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i] == shapes[j]:
                src.extend([i, j])
                dst.extend([j, i])
                etype.extend([1, 1])
    # relation 2: same color
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                src.extend([i, j])
                dst.extend([j, i])
                etype.extend([2, 2])
    if len(src) == 0:
        src, dst, etype = [0], [0], [0]

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
# label map (need only labels, token vocab not used here) -----------------
label2idx = {}
for r in train_rows + dev_rows + test_rows:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print(f"Num classes = {num_classes}")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# build graphs & loaders --------------------------------------------------
train_graphs = [seq_to_graph_tokenblind(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph_tokenblind(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph_tokenblind(r["sequence"], r["label"]) for r in test_rows]

batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# TOKEN-BLIND RGCN model --------------------------------------------------
class SPR_RGCN_TokenBlind(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64, num_rel=3, num_cls=2):
        super().__init__()
        # single learnable vector shared by all nodes
        self.embed = nn.Embedding(1, embed_dim)
        self.conv1 = RGCNConv(embed_dim, hidden_dim, num_relations=num_rel)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_rel)
        self.lin = nn.Linear(hidden_dim, num_cls)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.embed(x)  # x values are all 0, so returns shared vector
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = torch.relu(self.conv2(x, edge_index, edge_type))
        g_emb = global_mean_pool(x, batch)
        return self.lin(g_emb)


# -------------------------------------------------------------------------
# training utilities ------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, seqs, y_true, y_pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch.y.squeeze().cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cpx = complexity_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, cwa, swa, cpx, y_pred, y_true, seqs


# -------------------------------------------------------------------------
# experiment tracking dict -----------------------------------------------
experiment_data = {
    "ConstantNodeFeatures": {
        "SPR": {
            "metrics": {
                "train": {"CWA": [], "SWA": [], "CmpWA": []},
                "val": {"CWA": [], "SWA": [], "CmpWA": []},
            },
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# training loop -----------------------------------------------------------
max_epochs, patience = 40, 7
model = SPR_RGCN_TokenBlind(num_cls=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val, best_state, wait = float("inf"), None, 0

for epoch in range(1, max_epochs + 1):
    tr_loss, tr_cwa, tr_swa, tr_cpx, _, _, _ = run_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_cwa, val_swa, val_cpx, _, _, _ = run_epoch(
        model, val_loader, criterion
    )
    ex = experiment_data["ConstantNodeFeatures"]["SPR"]
    ex["losses"]["train"].append(tr_loss)
    ex["losses"]["val"].append(val_loss)
    ex["metrics"]["train"]["CWA"].append(tr_cwa)
    ex["metrics"]["train"]["SWA"].append(tr_swa)
    ex["metrics"]["train"]["CmpWA"].append(tr_cpx)
    ex["metrics"]["val"]["CWA"].append(val_cwa)
    ex["metrics"]["val"]["SWA"].append(val_swa)
    ex["metrics"]["val"]["CmpWA"].append(val_cpx)
    ex["epochs"].append(epoch)

    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} "
        f"val_loss={val_loss:.4f} val_CmpWA={val_cpx:.4f}"
    )
    if val_loss < best_val - 1e-4:
        best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# -------------------------------------------------------------------------
# evaluation on test set --------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_cwa, test_swa, test_cpx, test_pred, test_true, _ = run_epoch(
    model, test_loader, criterion
)
print(
    f"TEST: loss={test_loss:.4f} CWA={test_cwa:.4f} "
    f"SWA={test_swa:.4f} CmpWA={test_cpx:.4f}"
)
exp = experiment_data["ConstantNodeFeatures"]["SPR"]
exp["predictions"] = test_pred
exp["ground_truth"] = test_true
exp["test_metrics"] = {
    "loss": test_loss,
    "CWA": test_cwa,
    "SWA": test_swa,
    "CmpWA": test_cpx,
}
# -------------------------------------------------------------------------
# save experiment data ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
