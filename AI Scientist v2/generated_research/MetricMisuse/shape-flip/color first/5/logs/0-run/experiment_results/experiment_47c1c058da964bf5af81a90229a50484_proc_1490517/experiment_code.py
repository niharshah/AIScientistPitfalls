import os, pathlib, random, time, collections, numpy as np, torch
from typing import List, Tuple

# ========= Device =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ========= Try loading SPR_BENCH or fallback synthetic =========
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import importlib.util, sys

        spec = importlib.util.find_spec("SPR")
        if spec is None:
            raise ImportError
        SPR = importlib.import_module("SPR")
        DATA_PATH = pathlib.Path(
            os.environ.get(
                "SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
            )
        )
        dset = SPR.load_spr_bench(DATA_PATH)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        raise IOError from e


def build_synthetic_dataset(n_train=512, n_val=128, n_test=128):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 8))
        )

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
    print("Loaded real SPR_BENCH.")
except IOError:
    print("Using synthetic dataset.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()


# ========= Vocabularies =========
def extract_tokens(rows):
    for r in rows:
        for tok in r["sequence"].split():
            yield tok


token2idx = {"<PAD>": 0}
for tok in extract_tokens(train_rows + dev_rows + test_rows):
    token2idx.setdefault(tok, len(token2idx))
label2idx = {}
for r in train_rows + dev_rows + test_rows:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print("Vocab", len(token2idx), "labels", num_classes)

# ========= Build PyG graphs =========
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


# ========= Metrics =========
def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return (sum(c) / sum(w)) if sum(w) > 0 else 0.0


# ========= Dynamic GNN Model =========
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SPR_GNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_classes: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(embed_dim, hidden_dim))
        else:
            self.convs.append(GCNConv(embed_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ========= Training helpers =========
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, opt=None):
    training = opt is not None
    model.train() if training else model.eval()
    total_loss, seqs, ys, ps = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        if training:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).detach().cpu().tolist()
        ps.extend(pred)
        ys.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
    avg = total_loss / len(loader.dataset)
    cpx = complexity_weighted_accuracy(seqs, ys, ps)
    return avg, cpx, ps, ys, seqs


# ========= Hyperparameter sweep =========
depths = [1, 2, 3, 4, 5]
epochs = 5
experiment_data = {
    "num_gnn_layers": {
        "SPR": {
            "depths": depths,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "best_depth": None,
        }
    }
}

best_val = -1.0
best_state = None
best_depth = None

for depth in depths:
    print(f"\n=== Training depth={depth} ===")
    model = SPR_GNN(len(token2idx), n_classes=num_classes, num_layers=depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_cpx, _, _, _ = run_epoch(model, train_loader, opt=optimizer)
        val_loss, val_cpx, _, _, _ = run_epoch(model, val_loader)
        if ep == epochs:  # only log final epoch per depth
            experiment_data["num_gnn_layers"]["SPR"]["losses"]["train"].append(tr_loss)
            experiment_data["num_gnn_layers"]["SPR"]["losses"]["val"].append(val_loss)
            experiment_data["num_gnn_layers"]["SPR"]["metrics"]["train"].append(tr_cpx)
            experiment_data["num_gnn_layers"]["SPR"]["metrics"]["val"].append(val_cpx)
        print(
            f"Depth {depth} Ep {ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} train_CpxWA={tr_cpx:.4f} val_CpxWA={val_cpx:.4f} time={time.time()-t0:.1f}s"
        )
    if val_cpx > best_val:
        best_val = val_cpx
        best_state = model.state_dict()
        best_depth = depth
    torch.cuda.empty_cache()

print(f"\nBest depth={best_depth} with val_CpxWA={best_val:.4f}")
experiment_data["num_gnn_layers"]["SPR"]["best_depth"] = best_depth

# ========= Test evaluation with best model =========
best_model = SPR_GNN(len(token2idx), n_classes=num_classes, num_layers=best_depth).to(
    device
)
best_model.load_state_dict(best_state)
test_loss, test_cpx, test_pred, test_true, test_seq = run_epoch(best_model, test_loader)
print(f"Test: loss={test_loss:.4f} CpxWA={test_cpx:.4f}")

experiment_data["num_gnn_layers"]["SPR"]["losses"]["test"] = test_loss
experiment_data["num_gnn_layers"]["SPR"]["metrics"]["test"] = test_cpx
experiment_data["num_gnn_layers"]["SPR"]["predictions"] = test_pred
experiment_data["num_gnn_layers"]["SPR"]["ground_truth"] = test_true

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
