import os, pathlib, random, copy, numpy as np, torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn

# --------------------------------------------------------
# experiment-tracking dict (guideline-compliant structure)
# --------------------------------------------------------
experiment_data = {
    "NoSeqEdges": {
        "dataset": {
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

# --------------------------------------------------------
# env / device
# --------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------------
# try to load real dataset (SPR_BENCH); otherwise synthetic
# --------------------------------------------------------
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
    dataset_name = "SPR"
    print("Loaded real SPR_BENCH.")
except IOError:
    print("Using synthetic data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()
    dataset_name = "Synthetic"


# --------------------------------------------------------
# vocab / label maps
# --------------------------------------------------------
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
print(f"Vocab size={len(token2idx)}, num_classes={num_classes}")


# --------------------------------------------------------
# metric helpers
# --------------------------------------------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# --------------------------------------------------------
# NO-SEQUENTIAL-EDGE graph construction
# keeps only relations 1 (same shape) & 2 (same color)
# --------------------------------------------------------
def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    shapes = [t[0] for t in toks]
    colors = [t[1] if len(t) > 1 else "_" for t in toks]
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)

    src, dst, etype = [], [], []
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

    # ensure at least one edge (self-loop, use dummy relation 1)
    if not src:
        src, dst, etype = [0], [0], [1]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    return Data(
        x=node_feats,
        edge_index=edge_index,
        edge_type=edge_type,
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


# --------------------------------------------------------
# RGCN model (unchanged, num_rel=3)
# --------------------------------------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, vocab, embed_dim=64, hidden_dim=64, num_rel=3, num_cls=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.conv1 = RGCNConv(embed_dim, hidden_dim, num_relations=num_rel)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_rel)
        self.lin = nn.Linear(hidden_dim, num_cls)

    def forward(self, x, edge_index, edge_type, batch):
        h = self.embed(x)
        h = torch.relu(self.conv1(h, edge_index, edge_type))
        h = torch.relu(self.conv2(h, edge_index, edge_type))
        g = global_mean_pool(h, batch)
        return self.lin(g)


# --------------------------------------------------------
# train / eval routines
# --------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, seqs, y_true, y_pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        if train:
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


# --------------------------------------------------------
# training loop
# --------------------------------------------------------
max_epochs, patience = 40, 7
model = SPR_RGCN(len(token2idx), num_cls=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val, best_state, wait = float("inf"), None, 0
for epoch in range(1, max_epochs + 1):
    tr = run_epoch(model, train_loader, criterion, optimizer)
    vl = run_epoch(model, val_loader, criterion)
    # log
    experiment_data["NoSeqEdges"]["dataset"]["losses"]["train"].append(tr[0])
    experiment_data["NoSeqEdges"]["dataset"]["losses"]["val"].append(vl[0])
    for k, idx in zip(("CWA", "SWA", "CmpWA"), (1, 2, 3)):
        experiment_data["NoSeqEdges"]["dataset"]["metrics"]["train"][k].append(tr[idx])
        experiment_data["NoSeqEdges"]["dataset"]["metrics"]["val"][k].append(vl[idx])
    experiment_data["NoSeqEdges"]["dataset"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch:02d} | train_loss={tr[0]:.4f} val_loss={vl[0]:.4f} val_CmpWA={vl[3]:.4f}"
    )

    if vl[0] < best_val - 1e-4:
        best_val, best_state, wait = vl[0], copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# --------------------------------------------------------
# evaluate on test set
# --------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test = run_epoch(model, test_loader, criterion)
print(
    f"TEST | loss={test[0]:.4f} CWA={test[1]:.4f} SWA={test[2]:.4f} CmpWA={test[3]:.4f}"
)

experiment_data["NoSeqEdges"]["dataset"]["predictions"] = test[4]
experiment_data["NoSeqEdges"]["dataset"]["ground_truth"] = test[5]
experiment_data["NoSeqEdges"]["dataset"]["test_metrics"] = {
    "loss": test[0],
    "CWA": test[1],
    "SWA": test[2],
    "CmpWA": test[3],
}

# --------------------------------------------------------
# save results
# --------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
