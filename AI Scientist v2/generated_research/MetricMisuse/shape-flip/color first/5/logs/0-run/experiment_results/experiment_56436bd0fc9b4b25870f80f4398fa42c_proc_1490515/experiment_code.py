import os, pathlib, random, time, numpy as np, torch, collections
from typing import List, Tuple

# --------------------- housekeeping ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# ---------------------  dataset utils -------------------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import importlib.util, sys, SPR

        DATA_PATH = pathlib.Path(
            os.environ.get(
                "SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
            )
        )
        dset = SPR.load_spr_bench(DATA_PATH)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        raise IOError from e


def build_synth(n_train=512, n_val=128, n_test=128):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def mk(n):
        out = []
        for i in range(n):
            L = random.randint(4, 8)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            out.append({"id": i, "sequence": seq, "label": random.choice(labels)})
        return out

    def reid(lst):
        for i, r in enumerate(lst):
            r["id"] = i
        return lst

    return reid(mk(n_train)), reid(mk(n_val)), reid(mk(n_test))


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    print("Using synthetic dataset.")
    train_rows, dev_rows, test_rows = build_synth()


# ------------------- vocabulary -------------------------
def extract_tokens(rows):
    for r in rows:
        for t in r["sequence"].split():
            yield t


token2idx = {"<PAD>": 0}
for t in extract_tokens(train_rows + dev_rows + test_rows):
    token2idx.setdefault(t, len(token2idx))
label2idx = {}
for r in train_rows + dev_rows + test_rows:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print(f"Vocab size:{len(token2idx)}, classes:{num_classes}")

# ------------------- graph building ---------------------
from torch_geometric.data import Data


def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    if n > 1:
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
    else:
        src, dst = [0], [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=node_feats, edge_index=edge_index, y=y, seq=seq)


train_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in test_rows]

from torch_geometric.loader import DataLoader

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ------------------- metrics ----------------------------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split()))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [w_ if yt == yp else 0 for w_, yt, yp in zip(w, y_true, y_pred)]
    return (sum(corr) / sum(w)) if sum(w) > 0 else 0.0


# ------------------- model ------------------------------
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SPR_GNN(nn.Module):
    def __init__(self, vocab, embed_dim=64, hid=64, n_cl=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.conv1 = GCNConv(embed_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = nn.Linear(hid, n_cl)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ------------------- training loop ----------------------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss, seqs, y_t, y_p = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).detach().cpu().tolist()
        y_p.extend(preds)
        y_t.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = tot_loss / len(loader.dataset)
    cpx = complexity_weighted_accuracy(seqs, y_t, y_p)
    return avg_loss, cpx, y_p, y_t, seqs


# ---------------- hyperparameter sweep -----------------
param_grid = [1e-4, 3e-4, 1e-3, 3e-3]
epochs = 5
experiment_data = {"learning_rate": {}}

for lr in param_grid:
    print(f"\n=== Training with lr={lr} ===")
    model = SPR_GNN(len(token2idx), n_cl=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "lr": lr,
    }
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_cpx, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_cpx, _, _, _ = run_epoch(model, val_loader, criterion)
        exp_entry["losses"]["train"].append(tr_loss)
        exp_entry["losses"]["val"].append(val_loss)
        exp_entry["metrics"]["train"].append(tr_cpx)
        exp_entry["metrics"]["val"].append(val_cpx)
        exp_entry["epochs"].append(epoch)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_CpxWA={tr_cpx:.4f} val_CpxWA={val_cpx:.4f} "
            f"elapsed={time.time()-t0:.1f}s"
        )
    # test
    test_loss, test_cpx, test_pred, test_true, test_seq = run_epoch(
        model, test_loader, criterion
    )
    exp_entry["losses"]["test"] = test_loss
    exp_entry["metrics"]["test"] = test_cpx
    exp_entry["predictions"] = test_pred
    exp_entry["ground_truth"] = test_true
    print(f"Test: loss={test_loss:.4f} CpxWA={test_cpx:.4f}")
    experiment_data["learning_rate"][f"lr={lr}"] = {"SPR": exp_entry}

# ------------------- save -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved results to {os.path.join(working_dir,'experiment_data.npy')}")
