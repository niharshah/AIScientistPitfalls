import os, pathlib, copy, random, time, numpy as np, torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn as nn

# ---------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# 1) Try to load real benchmark -----------------------------------------------------------------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import importlib.util, sys

        spec = importlib.util.find_spec("SPR")
        if spec is None:
            raise ImportError
        SPR = importlib.import_module("SPR")
        root = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(root)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        raise IOError from e


def build_synth_split(n: int, offset: int = 0):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]
    rows = []
    for i in range(n):
        L = random.randint(4, 8)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        rows.append({"id": offset + i, "sequence": seq, "label": random.choice(labels)})
    return rows


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    print("Loaded real SPR_BENCH.")
except IOError:
    print("Real dataset not found – generating synthetic toy data.")
    train_rows, dev_rows, test_rows = (
        build_synth_split(512),
        build_synth_split(128, 512),
        build_synth_split(128, 640),
    )


# ---------------------------------------------------------------------------
# 2) Vocabulary -----------------------------------------------------------------
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
print(f"Vocab={len(token2idx)}  #labels={num_classes}")


# ---------------------------------------------------------------------------
# 3) Graph construction with 3 relation types ----------------------------------
#  relation 0: order (i<->i+1)
#  relation 1: same color
#  relation 2: same shape
def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    # --- nodes
    x = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    edge_src, edge_dst, edge_type = [], [], []
    # order edges
    for i in range(n - 1):
        edge_src += [i, i + 1]
        edge_dst += [i + 1, i]
        edge_type += [0, 0]
    # color & shape relations
    colors = [t[1] for t in toks]
    shapes = [t[0] for t in toks]
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                edge_src += [i, j]
                edge_dst += [j, i]
                edge_type += [1, 1]
            if shapes[i] == shapes[j]:
                edge_src += [i, j]
                edge_dst += [j, i]
                edge_type += [2, 2]
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


train_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in test_rows]

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ---------------------------------------------------------------------------
# 4) Metrics -------------------------------------------------------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split()})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    cor = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    cor = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    cor = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


# ---------------------------------------------------------------------------
# 5) Model ---------------------------------------------------------------------
class RGCNClassifier(nn.Module):
    def __init__(self, vocab_size, embed=64, hidden=64, num_rel=3, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.conv1 = RGCNConv(embed, hidden, num_rel)
        self.conv2 = RGCNConv(hidden, hidden, num_rel)
        self.lin = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.emb(x)
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = torch.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------------------------------------------------------------------------
# 6) Training helpers ----------------------------------------------------------
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
        yp.extend(preds)
        ys.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
    n = len(loader.dataset)
    avg_loss = tot_loss / n
    return avg_loss, seqs, ys, yp


def compute_metrics(seqs, ys, yp):
    return (
        color_weighted_accuracy(seqs, ys, yp),
        shape_weighted_accuracy(seqs, ys, yp),
        complexity_weighted_accuracy(seqs, ys, yp),
    )


# ---------------------------------------------------------------------------
# 7) Experiment loop -----------------------------------------------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

max_epochs = 20
patience = 3
model = RGCNClassifier(len(token2idx), n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val = float("inf")
best_state = None
wait = 0
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_seqs, tr_ys, tr_yp = run_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_seqs, val_ys, val_yp = run_epoch(model, val_loader, criterion)
    tr_cwa, tr_swa, tr_cpx = compute_metrics(tr_seqs, tr_ys, tr_yp)
    val_cwa, val_swa, val_cpx = compute_metrics(val_seqs, val_ys, val_yp)
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["train"].append(
        {"CWA": tr_cwa, "SWA": tr_swa, "CmpWA": tr_cpx}
    )
    experiment_data["SPR"]["metrics"]["val"].append(
        {"CWA": val_cwa, "SWA": val_swa, "CmpWA": val_cpx}
    )
    experiment_data["SPR"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | Val CWA={val_cwa:.3f} SWA={val_swa:.3f} CmpWA={val_cpx:.3f}"
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

# ---------------------------------------------------------------------------
# 8) Test evaluation -----------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_seqs, test_ys, test_yp = run_epoch(model, test_loader, criterion)
test_cwa, test_swa, test_cpx = compute_metrics(test_seqs, test_ys, test_yp)
print(
    f"Test   loss={test_loss:.4f}  CWA={test_cwa:.3f}  SWA={test_swa:.3f}  CmpWA={test_cpx:.3f}"
)

experiment_data["SPR"]["predictions"] = test_yp
experiment_data["SPR"]["ground_truth"] = test_ys
experiment_data["SPR"]["test"] = {
    "loss": test_loss,
    "CWA": test_cwa,
    "SWA": test_swa,
    "CmpWA": test_cpx,
}

# ---------------------------------------------------------------------------
# 9) Save experiment data ------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
