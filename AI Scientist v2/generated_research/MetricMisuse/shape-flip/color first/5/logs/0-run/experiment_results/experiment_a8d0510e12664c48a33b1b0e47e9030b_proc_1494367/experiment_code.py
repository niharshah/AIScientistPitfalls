import os, pathlib, random, time, copy, numpy as np, torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn
from typing import List, Tuple

# ------------------ boilerplate & device ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ dataset loading ------------------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import SPR, importlib.util

        DATA_PATH = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(DATA_PATH)
        return dset["train"], dset["dev"], dset["test"]
    except Exception:
        raise IOError


def build_synthetic_dataset(n_train=800, n_val=200, n_test=200):
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
    print("Loaded real SPR_BENCH.")
except IOError:
    print("Using synthetic SPR data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()

# ------------------ vocab & label maps ------------------
token2idx = {"<PAD>": 0}
for r in train_rows + dev_rows + test_rows:
    for tok in r["sequence"].split():
        if tok not in token2idx:
            token2idx[tok] = len(token2idx)
label2idx = {}
for r in train_rows + dev_rows + test_rows:
    if r["label"] not in label2idx:
        label2idx[r["label"]] = len(label2idx)
num_classes = len(label2idx)
print(f"Vocab={len(token2idx)}, Classes={num_classes}")


# ------------------ helpers for metrics ------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split()})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def cmpwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


# ------------------ graph construction with 3 relations ------------------
REL_ADJ, REL_COLOR, REL_SHAPE = 0, 1, 2


def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    x = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    src, dst, etype = [], [], []
    # adjacency edges
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [REL_ADJ, REL_ADJ]
    # same color edges
    color_map = {}
    for i, t in enumerate(toks):
        color_map.setdefault(t[1], []).append(i)
    for nodes in color_map.values():
        for i in nodes:
            for j in nodes:
                if i != j:
                    src.append(i)
                    dst.append(j)
                    etype.append(REL_COLOR)
    # same shape edges
    shape_map = {}
    for i, t in enumerate(toks):
        shape_map.setdefault(t[0], []).append(i)
    for nodes in shape_map.values():
        for i in nodes:
            for j in nodes:
                if i != j:
                    src.append(i)
                    dst.append(j)
                    etype.append(REL_SHAPE)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label2idx[label]], dtype=torch.long),
        seq=seq,
    )


train_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in test_rows]
batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ------------------ model ------------------
class SPR_RGCN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64, num_rel=3, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = RGCNConv(embed_dim, hidden, num_rel)
        self.conv2 = RGCNConv(hidden, hidden, num_rel)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = torch.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ------------------ training utils ------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, seqs, ys, yp = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        yp += preds
        ys += batch.y.cpu().tolist()
        seqs += batch.seq
    avg = tot_loss / len(loader.dataset)
    return avg, cwa(seqs, ys, yp), swa(seqs, ys, yp), cmpwa(seqs, ys, yp), yp, ys, seqs


# ------------------ experiment ------------------
max_epochs = 40
patience = 5
model = SPR_RGCN(len(token2idx), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = {
    "losses": {"train": [], "val": []},
    "metrics": {
        "CWA": {"train": [], "val": []},
        "SWA": {"train": [], "val": []},
        "CmpWA": {"train": [], "val": []},
    },
    "epochs": [],
}
best_val, best_state = float("inf"), None
wait = 0
for ep in range(1, max_epochs + 1):
    tr_loss, tr_cwa, tr_swa, tr_cmp, _, _, _ = run_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_cwa, val_swa, val_cmp, _, _, _ = run_epoch(
        model, val_loader, criterion
    )
    history["losses"]["train"].append(tr_loss)
    history["losses"]["val"].append(val_loss)
    history["metrics"]["CWA"]["train"].append(tr_cwa)
    history["metrics"]["CWA"]["val"].append(val_cwa)
    history["metrics"]["SWA"]["train"].append(tr_swa)
    history["metrics"]["SWA"]["val"].append(val_swa)
    history["metrics"]["CmpWA"]["train"].append(tr_cmp)
    history["metrics"]["CmpWA"]["val"].append(val_cmp)
    history["epochs"].append(ep)
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f} | val_CmpWA={val_cmp:.4f}")
    if val_loss < best_val - 1e-4:
        best_val, val_cmp_best = val_loss, val_cmp
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
model.load_state_dict(best_state)
test_loss, test_cwa, test_swa, test_cmp, ypred, ytrue, seqs = run_epoch(
    model, test_loader, criterion
)
print(
    f"TEST  loss={test_loss:.4f}  CWA={test_cwa:.4f}  SWA={test_swa:.4f}  CmpWA={test_cmp:.4f}"
)

# ------------------ save experiment ------------------
experiment_data = {
    "SPR_dataset": {
        "metrics": history["metrics"],
        "losses": history["losses"],
        "test": {
            "loss": test_loss,
            "CWA": test_cwa,
            "SWA": test_swa,
            "CmpWA": test_cmp,
            "predictions": ypred,
            "ground_truth": ytrue,
        },
        "epochs": history["epochs"],
    }
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
