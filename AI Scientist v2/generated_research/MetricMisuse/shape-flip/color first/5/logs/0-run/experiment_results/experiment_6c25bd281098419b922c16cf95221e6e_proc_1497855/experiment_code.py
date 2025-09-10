# Shallow-GNN (1-hop) ablation study – self-contained runnable script
import os, pathlib, random, copy, numpy as np, torch, time
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------------------------------------------------
# dataset loading (real SPR_BENCH if available, else synthetic fallback)
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import SPR

        DATA_PATH = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        ds = SPR.load_spr_bench(DATA_PATH)
        return ds["train"], ds["dev"], ds["test"]
    except Exception:
        raise IOError


def build_synthetic_dataset(n_train=600, n_val=150, n_test=150):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 10))
        )

    def make_split(n):
        lst = [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]
        for i, r in enumerate(lst):
            r["id"] = i
        return lst

    return make_split(n_train), make_split(n_val), make_split(n_test)


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    dataset_name = "SPR_BENCH"
    print("Loaded real SPR_BENCH.")
except IOError:
    train_rows, dev_rows, test_rows = build_synthetic_dataset()
    dataset_name = "synthetic"
    print("Using synthetic dataset.")


# -------------------------------------------------------------------------
# vocab / label maps
def all_tokens(rows):
    for r in rows:
        for t in r["sequence"].split():
            yield t


token2idx = {"<PAD>": 0}
for tok in all_tokens(train_rows + dev_rows + test_rows):
    token2idx.setdefault(tok, len(token2idx))

label2idx = {}
for r in train_rows + dev_rows + test_rows:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print(f"Vocab={len(token2idx)}  Labels={num_classes}")


# -------------------------------------------------------------------------
# metrics
def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def _w_acc(seqs, y_t, y_p, func):
    w = [func(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(a, b, c):
    return _w_acc(a, b, c, count_color_variety)


def shape_weighted_accuracy(a, b, c):
    return _w_acc(a, b, c, count_shape_variety)


def complexity_weighted_accuracy(a, b, c):
    return _w_acc(a, b, c, lambda s: count_color_variety(s) + count_shape_variety(s))


# -------------------------------------------------------------------------
# sequence → heterogeneous graph
def seq_to_graph(seq, label) -> Data:
    toks = seq.split()
    n = len(toks)
    shapes = [t[0] for t in toks]
    colors = [t[1] if len(t) > 1 else "_" for t in toks]
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    src, dst, etype = [], [], []
    # relation 0: order
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
    if not src:
        src, dst, etype = [0], [0], [0]  # self-loop fallback
    return Data(
        x=node_feats,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.tensor(etype, dtype=torch.long),
        y=torch.tensor([label2idx[label]], dtype=torch.long),
        seq=seq,
    )


train_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in test_rows]

batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# -------------------------------------------------------------------------
# Shallow 1-hop RGCN model
class Shallow_RGCN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_rel=3, num_cls=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = RGCNConv(embed_dim, hidden_dim, num_relations=num_rel)
        self.lin = nn.Linear(hidden_dim, num_cls)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        g_emb = global_mean_pool(x, batch)
        return self.lin(g_emb)


# -------------------------------------------------------------------------
# training utilities
def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    tot_loss, seqs, y_t, y_p = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        if train_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        y_p.extend(pred)
        y_t.extend(batch.y.squeeze().cpu().tolist())
        seqs.extend(batch.seq)
    N = len(loader.dataset)
    return (
        tot_loss / N,
        color_weighted_accuracy(seqs, y_t, y_p),
        shape_weighted_accuracy(seqs, y_t, y_p),
        complexity_weighted_accuracy(seqs, y_t, y_p),
        y_p,
        y_t,
        seqs,
    )


# -------------------------------------------------------------------------
# experiment tracking dict
experiment_data = {
    "Shallow_GNN_1hop": {
        dataset_name: {
            "metrics": {
                "train": {"CWA": [], "SWA": [], "CmpWA": []},
                "val": {"CWA": [], "SWA": [], "CmpWA": []},
            },
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
            "test_metrics": {},
        }
    }
}

# -------------------------------------------------------------------------
# training loop
max_epochs = 40
patience = 7
model = Shallow_RGCN(len(token2idx), num_cls=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val = float("inf")
best_state = None
wait = 0
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_cwa, tr_swa, tr_cpx, _, _, _ = run_epoch(
        model, train_loader, criterion, optimizer
    )
    vl_loss, vl_cwa, vl_swa, vl_cpx, _, _, _ = run_epoch(model, val_loader, criterion)
    ed = experiment_data["Shallow_GNN_1hop"][dataset_name]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(vl_loss)
    ed["metrics"]["train"]["CWA"].append(tr_cwa)
    ed["metrics"]["train"]["SWA"].append(tr_swa)
    ed["metrics"]["train"]["CmpWA"].append(tr_cpx)
    ed["metrics"]["val"]["CWA"].append(vl_cwa)
    ed["metrics"]["val"]["SWA"].append(vl_swa)
    ed["metrics"]["val"]["CmpWA"].append(vl_cpx)
    ed["epochs"].append(epoch)
    print(
        f"Epoch {epoch:02d}  train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  val_CmpWA={vl_cpx:.4f}"
    )
    if vl_loss < best_val - 1e-4:
        best_val = vl_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# -------------------------------------------------------------------------
# evaluation on test
if best_state is not None:
    model.load_state_dict(best_state)
ts_loss, ts_cwa, ts_swa, ts_cpx, ts_pred, ts_true, _ = run_epoch(
    model, test_loader, criterion
)
print(
    f"TEST  loss={ts_loss:.4f}  CWA={ts_cwa:.4f}  SWA={ts_swa:.4f}  CmpWA={ts_cpx:.4f}"
)

ed = experiment_data["Shallow_GNN_1hop"][dataset_name]
ed["predictions"] = ts_pred
ed["ground_truth"] = ts_true
ed["test_metrics"] = {"loss": ts_loss, "CWA": ts_cwa, "SWA": ts_swa, "CmpWA": ts_cpx}

# -------------------------------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
