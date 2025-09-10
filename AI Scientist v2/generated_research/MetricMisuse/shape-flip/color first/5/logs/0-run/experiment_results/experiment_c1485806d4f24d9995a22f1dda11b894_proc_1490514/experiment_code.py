import os, pathlib, random, time, collections, numpy as np, torch
from typing import List, Tuple

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset ----------
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import importlib.util, sys

        spec = importlib.util.find_spec("SPR")
        SPR = importlib.import_module("SPR") if spec else None
        DATA_PATH = pathlib.Path(
            os.environ.get(
                "SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
            )
        )
        dset = SPR.load_spr_bench(DATA_PATH) if SPR else None
        return dset["train"], dset["dev"], dset["test"]
    except Exception:
        raise IOError


def build_synthetic_dataset(n_train=512, n_val=128, n_test=128):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]
    make_seq = lambda: " ".join(
        random.choice(shapes) + random.choice(colors)
        for _ in range(random.randint(4, 8))
    )

    def make_split(n):
        return [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]

    def build_id(lst):
        [lst.__setitem__(i, {**row, "id": i}) for i, row in enumerate(lst)]
        return lst

    return (*(build_id(make_split(n)) for n in (n_train, n_val, n_test)),)


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    print("Could not load real dataset – using synthetic data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()


# ---------- vocab ----------
def extract_tokens(rows):
    return (tok for r in rows for tok in r["sequence"].split())


token2idx = {"<PAD>": 0}
[
    token2idx.setdefault(tok, len(token2idx))
    for tok in extract_tokens(train_rows + dev_rows + test_rows)
]
label2idx = {}
[
    label2idx.setdefault(r["label"], len(label2idx))
    for r in train_rows + dev_rows + test_rows
]
num_classes = len(label2idx)
print(f"Vocab size={len(token2idx)}, #labels={num_classes}")

# ---------- graphs ----------
from torch_geometric.data import Data


def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    src = list(range(n - 1)) + list(range(1, n)) if n > 1 else [0]
    dst = list(range(1, n)) + list(range(n - 1)) if n > 1 else [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=node_feats, edge_index=edge_index, y=y, seq=seq)


train_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in test_rows]

from torch_geometric.loader import DataLoader

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ---------- metrics ----------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


# ---------- model ----------
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SPR_GNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1, self.conv2 = GCNConv(embed_dim, hidden_dim), GCNConv(
            hidden_dim, hidden_dim
        )
        self.lin = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------- training helpers ----------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, seqs, true, pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).detach().cpu().tolist()
        pred.extend(preds)
        true.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
    return (
        total_loss / len(loader.dataset),
        complexity_weighted_accuracy(seqs, true, pred),
        pred,
        true,
        seqs,
    )


# ---------- hyperparameter tuning ----------
embed_grid = [32, 64, 128, 256]
experiment_data = {
    "embed_dim": {
        "SPR": {
            "embed_dims": embed_grid,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "best_embed": None,
            "predictions": [],
            "ground_truth": [],
            "epochs": list(range(1, 6)),
        }
    }
}

best_val = -1
best_state = None
best_embed = None
for ed in embed_grid:
    torch.cuda.empty_cache()
    model = SPR_GNN(len(token2idx), embed_dim=ed, n_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_metrics_per_epoch = []
    val_metrics_per_epoch = []
    train_losses_per_epoch = []
    val_losses_per_epoch = []
    for epoch in range(1, 6):
        tr_loss, tr_cpx, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_cpx, _, _, _ = run_epoch(model, val_loader, criterion)
        train_losses_per_epoch.append(tr_loss)
        val_losses_per_epoch.append(vl_loss)
        train_metrics_per_epoch.append(tr_cpx)
        val_metrics_per_epoch.append(vl_cpx)
    experiment_data["embed_dim"]["SPR"]["metrics"]["train"].append(
        train_metrics_per_epoch
    )
    experiment_data["embed_dim"]["SPR"]["metrics"]["val"].append(val_metrics_per_epoch)
    experiment_data["embed_dim"]["SPR"]["losses"]["train"].append(
        train_losses_per_epoch
    )
    experiment_data["embed_dim"]["SPR"]["losses"]["val"].append(val_losses_per_epoch)
    if val_metrics_per_epoch[-1] > best_val:
        best_val = val_metrics_per_epoch[-1]
        best_state = model.state_dict()
        best_embed = ed
        best_model = model  # keep reference to evaluate test
print(f"Best embed_dim={best_embed} with val CpxWA={best_val:.4f}")
experiment_data["embed_dim"]["SPR"]["best_embed"] = best_embed

# ---------- test evaluation ----------
best_model.eval()
test_loss, test_cpx, test_pred, test_true, test_seq = run_epoch(
    best_model, test_loader, criterion
)
print(f"Test: loss={test_loss:.4f}, CpxWA={test_cpx:.4f}")
experiment_data["embed_dim"]["SPR"]["predictions"] = test_pred
experiment_data["embed_dim"]["SPR"]["ground_truth"] = test_true

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
