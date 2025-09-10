import os, pathlib, random, time, collections, numpy as np, torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

# ============ I/O prep ============
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============ Dataset loading / synthetic fallback ============
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import importlib.util, importlib

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
        L = random.randint(4, 8)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def make_split(n):
        return [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]

    def assign_id(lst):
        for i, row in enumerate(lst):
            row["id"] = i
        return lst

    return tuple(assign_id(make_split(n)) for n in (n_train, n_val, n_test))


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    print("Could not load real dataset – using synthetic data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()

# ============ Vocabulary ============
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
print(f"Vocab size={len(token2idx)}, #labels={num_classes}")


# ============ Graph construction ============
def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.strip().split()
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

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ============ Metrics ============
def count_color_variety(seq):  # number of distinct colors
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):  # number of distinct shapes
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return (sum(correct) / sum(w)) if sum(w) else 0.0


# ============ Model ============
class SPR_GNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ============ Training utilities ============
def run_epoch(loader, model, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_seqs, all_true, all_pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).detach().cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(batch.y.detach().cpu().tolist())
        all_seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    cpx_acc = complexity_weighted_accuracy(all_seqs, all_true, all_pred)
    return avg_loss, cpx_acc, all_pred, all_true, all_seqs


# ============ Hyperparameter sweep ============
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
epochs = 5
experiment_data = {"weight_decay": {"SPR": {}}}

for wd in weight_decays:
    tag = f"wd_{wd}"
    print(f"\n=== Training with weight_decay={wd} ===")
    model = SPR_GNN(len(token2idx), n_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    logs = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_cpx, *_ = run_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_cpx, _, _, _ = run_epoch(val_loader, model, criterion)

        logs["losses"]["train"].append(tr_loss)
        logs["losses"]["val"].append(val_loss)
        logs["metrics"]["train"].append(tr_cpx)
        logs["metrics"]["val"].append(val_cpx)
        logs["epochs"].append(epoch)

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_CpxWA={tr_cpx:.4f}, val_CpxWA={val_cpx:.4f}, "
            f"elapsed={time.time()-t0:.1f}s"
        )

    # test evaluation
    test_loss, test_cpx, test_pred, test_true, _ = run_epoch(
        test_loader, model, criterion
    )
    print(f"Test: loss={test_loss:.4f}, CpxWA={test_cpx:.4f}")
    logs["test_loss"] = test_loss
    logs["test_metric"] = test_cpx
    logs["predictions"] = test_pred
    logs["ground_truth"] = test_true

    experiment_data["weight_decay"]["SPR"][tag] = logs

# ============ Save ============
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy to", working_dir)
