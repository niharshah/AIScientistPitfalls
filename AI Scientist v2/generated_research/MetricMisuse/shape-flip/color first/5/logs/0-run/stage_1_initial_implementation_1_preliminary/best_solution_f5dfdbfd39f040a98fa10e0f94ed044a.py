import os, pathlib, random, time, collections, numpy as np, torch
from typing import List, Tuple

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ============ Device handling ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============ Try to load real SPR_BENCH dataset ============
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Returns (train, dev, test) lists of dicts or raises IOError.
    Each dict has keys: id, sequence, label
    """
    try:
        import importlib.util, sys

        # try to import SPR.py located somewhere on PYTHONPATH
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


def build_synthetic_dataset(
    n_train=512, n_val=128, n_test=128
) -> Tuple[List[dict], List[dict], List[dict]]:
    shapes = ["C", "S", "T"]  # circle, square, triangle
    colors = ["r", "g", "b", "y"]
    labels = ["rule1", "rule2"]

    def make_seq():
        L = random.randint(4, 8)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
        return " ".join(toks)

    def make_split(n):
        return [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]

    return (
        build_id(make_split(n_train)),
        build_id(make_split(n_val)),
        build_id(make_split(n_test)),
    )


def build_id(lst):
    for i, row in enumerate(lst):
        row["id"] = i
    return lst


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    print("Could not load real dataset – using synthetic data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()


# ============ Token & Label vocabularies ============
def extract_tokens(rows):
    for r in rows:
        for tok in r["sequence"].split():
            yield tok


token2idx = {"<PAD>": 0}
for tok in extract_tokens(train_rows + dev_rows + test_rows):
    if tok not in token2idx:
        token2idx[tok] = len(token2idx)
label2idx = {}
for r in train_rows + dev_rows + test_rows:
    if r["label"] not in label2idx:
        label2idx[r["label"]] = len(label2idx)
num_classes = len(label2idx)
print(f"Vocab size = {len(token2idx)},  #labels = {num_classes}")

# ============ Build PyG graphs ============
from torch_geometric.data import Data


def seq_to_graph(seq: str, label: str) -> Data:
    tokens = seq.strip().split()
    n = len(tokens)
    node_feats = torch.tensor([token2idx[t] for t in tokens], dtype=torch.long)
    # Line graph edges i<->i+1
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

# ============ PyG DataLoaders ============
from torch_geometric.loader import DataLoader

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ============ Metric helpers ============
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return (sum(correct) / sum(weights)) if sum(weights) > 0 else 0.0


# ============ Simple GNN model ============
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SPR_GNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_classes: int = 2,
    ):
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


model = SPR_GNN(len(token2idx), n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============ Experiment data dict ============
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ============ Training loop ============
def run_epoch(loader, training: bool):
    if training:
        model.train()
    else:
        model.eval()
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
    cpx = complexity_weighted_accuracy(all_seqs, all_true, all_pred)
    return avg_loss, cpx, all_pred, all_true, all_seqs


epochs = 5
for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_loss, train_cpx, _, _, _ = run_epoch(train_loader, training=True)
    val_loss, val_cpx, val_pred, val_true, val_seq = run_epoch(
        val_loader, training=False
    )

    experiment_data["SPR"]["losses"]["train"].append(train_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["train"].append(train_cpx)
    experiment_data["SPR"]["metrics"]["val"].append(val_cpx)
    experiment_data["SPR"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: "
        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
        f"train_CpxWA={train_cpx:.4f}, val_CpxWA={val_cpx:.4f}, "
        f"elapsed={time.time()-t0:.1f}s"
    )

# ============ Test evaluation ============
test_loss, test_cpx, test_pred, test_true, test_seq = run_epoch(
    test_loader, training=False
)
print(f"\nTest   : loss={test_loss:.4f}, CpxWA={test_cpx:.4f}")

experiment_data["SPR"]["predictions"] = test_pred
experiment_data["SPR"]["ground_truth"] = test_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
