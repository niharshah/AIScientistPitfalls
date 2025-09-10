# ===================  num_epochs hyper-parameter tuning experiment  ===================
import os, pathlib, random, time, copy, numpy as np, torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

# ---------- misc ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- try to load real SPR_BENCH or make synthetic ----------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import importlib.util, sys

        spec = importlib.util.find_spec("SPR")
        if spec is None:
            raise ImportError
        SPR = importlib.import_module("SPR")
        DATA_PATH = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(DATA_PATH)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        raise IOError from e


def build_synthetic_dataset(n_train=512, n_val=128, n_test=128):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def make_seq():
        L = random.randint(4, 8)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def make_split(n):
        return [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]

    def build_id(lst):
        for i, r in enumerate(lst):
            r["id"] = i
        return lst

    return (
        build_id(make_split(n_train)),
        build_id(make_split(n_val)),
        build_id(make_split(n_test)),
    )


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    print("Could not load real dataset – using synthetic data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()


# ---------- vocab ----------
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
print(f"Vocab size={len(token2idx)}, #labels={num_classes}")


# ---------- build PyG graphs ----------
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
    return Data(
        x=node_feats,
        edge_index=edge_index,
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


# ---------- metrics ----------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split()})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return (sum(correct) / sum(w)) if sum(w) > 0 else 0.0


# ---------- model ----------
class SPR_GNN(nn.Module):
    def __init__(self, vocab, embed_dim=64, hidden=64, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.conv1 = GCNConv(embed_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------- helper: run one epoch ----------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss, seqs, ys, yp = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        yp.extend(preds)
        ys.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = tot_loss / len(loader.dataset)
    cpx_acc = complexity_weighted_accuracy(seqs, ys, yp)
    return avg_loss, cpx_acc, yp, ys, seqs


# ---------- experiment data ----------
experiment_data = {"num_epochs": {"SPR": {}}}

# ---------- hyper-parameter search ----------
epoch_options = [5, 20, 35, 50]
patience = 5
for max_epochs in epoch_options:
    print(f"\n=== Training with max_epochs={max_epochs} ===")
    model = SPR_GNN(len(token2idx), n_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    hist = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
    }

    best_state, best_val, wait = None, float("inf"), 0
    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_cpx, *_ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_cpx, *_ = run_epoch(model, val_loader, criterion)
        hist["losses"]["train"].append(tr_loss)
        hist["losses"]["val"].append(val_loss)
        hist["metrics"]["train"].append(tr_cpx)
        hist["metrics"]["val"].append(val_cpx)
        hist["epochs"].append(epoch)
        print(
            f"Ep {epoch:02d}/{max_epochs} "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_CpxWA={tr_cpx:.4f} val_CpxWA={val_cpx:.4f}"
        )
        # early stopping
        if val_loss < best_val - 1e-4:
            best_val, wait, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
    train_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # final evaluation
    test_loss, test_cpx, test_pred, test_true, _ = run_epoch(
        model, test_loader, criterion
    )
    print(
        f"Best model test_loss={test_loss:.4f} test_CpxWA={test_cpx:.4f} "
        f"(trained {len(hist['epochs'])} epochs, {train_time:.1f}s)"
    )

    hist["predictions"] = test_pred
    hist["ground_truth"] = test_true
    hist["test_loss"] = test_loss
    hist["test_CpxWA"] = test_cpx
    hist["train_time_s"] = train_time
    experiment_data["num_epochs"]["SPR"][f"epochs_{max_epochs}"] = hist

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
