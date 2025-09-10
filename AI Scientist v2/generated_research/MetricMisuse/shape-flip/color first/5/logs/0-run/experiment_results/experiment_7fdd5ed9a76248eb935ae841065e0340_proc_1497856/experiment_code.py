import os, pathlib, random, copy, numpy as np, torch, time
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# try to load official SPR dataset, otherwise build synthetic -------------
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
    dataset_name = "SPR-BENCH"
    print("Loaded SPR-BENCH.")
except IOError:
    print("Using synthetic data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()
    dataset_name = "synthetic"
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# vocab & label maps
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
vocab_size = len(token2idx)
print(f"Vocab={vocab_size}, Labels={num_classes}")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# metrics
def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


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


# -------------------------------------------------------------------------
# graph construction
def seq_to_graph(seq: str, label: str) -> Data:
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
    if len(src) == 0:
        src = [0]
        dst = [0]
        etype = [0]
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
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Model with frozen one-hot node features
class SPR_RGCN_OneHot(nn.Module):
    def __init__(self, vocab, hidden_dim=64, num_rel=3, num_cls=2):
        super().__init__()
        # first conv consumes one-hot directly
        self.conv1 = RGCNConv(vocab, hidden_dim, num_relations=num_rel)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_rel)
        self.lin = nn.Linear(hidden_dim, num_cls)
        self.vocab = vocab  # saved for one_hot

    def forward(self, x, edge_index, edge_type, batch):
        # convert indices to one-hot (non-learnable)
        x = F.one_hot(x, num_classes=self.vocab).float()
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = torch.relu(self.conv2(x, edge_index, edge_type))
        g_emb = global_mean_pool(x, batch)
        return self.lin(g_emb)


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# training utilities
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
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cpx = complexity_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, cwa, swa, cpx, y_pred, y_true, seqs


# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# experiment tracking dict
experiment_data = {
    "onehot_frozen": {
        dataset_name: {
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
store = experiment_data["onehot_frozen"][dataset_name]
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# train loop
max_epochs, patience = 40, 7
model = SPR_RGCN_OneHot(vocab_size, num_cls=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val = float("inf")
best_state = None
wait = 0
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_cwa, tr_swa, tr_cpx, _, _, _ = run_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_cwa, val_swa, val_cpx, _, _, _ = run_epoch(
        model, val_loader, criterion
    )
    # log
    store["losses"]["train"].append(tr_loss)
    store["losses"]["val"].append(val_loss)
    store["metrics"]["train"]["CWA"].append(tr_cwa)
    store["metrics"]["train"]["SWA"].append(tr_swa)
    store["metrics"]["train"]["CmpWA"].append(tr_cpx)
    store["metrics"]["val"]["CWA"].append(val_cwa)
    store["metrics"]["val"]["SWA"].append(val_swa)
    store["metrics"]["val"]["CmpWA"].append(val_cpx)
    store["epochs"].append(epoch)
    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_CmpWA={val_cpx:.4f}"
    )
    # early stopping
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# evaluate on test set
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_cwa, test_swa, test_cpx, test_pred, test_true, _ = run_epoch(
    model, test_loader, criterion
)
print(
    f"TEST: loss={test_loss:.4f} CWA={test_cwa:.4f} SWA={test_swa:.4f} CmpWA={test_cpx:.4f}"
)

store["predictions"] = test_pred
store["ground_truth"] = test_true
store["test_metrics"] = {
    "loss": test_loss,
    "CWA": test_cwa,
    "SWA": test_swa,
    "CmpWA": test_cpx,
}
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
