# Batch-size hyper-parameter tuning for SPR GNN
import os, pathlib, random, time, numpy as np, torch
from typing import List, Tuple

# ============ Reproducibility ============
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ============ Device ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============ Load dataset (real or synthetic) ============
def try_load_real_dataset() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import importlib

        spec = importlib.util.find_spec("SPR")
        if spec is None:
            raise ImportError
        SPR = importlib.import_module("SPR")
        data_path = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(data_path)
        return dset["train"], dset["dev"], dset["test"]
    except Exception:
        raise IOError


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

    def add_id(lst):
        for i, r in enumerate(lst):
            r["id"] = i
        return lst

    return (
        add_id(make_split(n_train)),
        add_id(make_split(n_val)),
        add_id(make_split(n_test)),
    )


try:
    train_rows, dev_rows, test_rows = try_load_real_dataset()
    print("Loaded real SPR_BENCH dataset.")
except IOError:
    print("Could not load real dataset – using synthetic data.")
    train_rows, dev_rows, test_rows = build_synthetic_dataset()


# ============ Vocabularies ============
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
print(f"Vocab size={len(token2idx)}, #labels={num_classes}")

# ============ Graph building ============
from torch_geometric.data import Data


def seq_to_graph(seq: str, label: str) -> Data:
    tokens = seq.split()
    n = len(tokens)
    node_feats = torch.tensor([token2idx[t] for t in tokens], dtype=torch.long)
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


# ============ Metrics ============
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split()))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return (sum(corr) / sum(w)) if sum(w) > 0 else 0.0


# ============ Model ============
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


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


# ============ Training helpers ============
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss, seqs, true, pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        ps = out.argmax(dim=-1).cpu().tolist()
        pred.extend(ps)
        true.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = tot_loss / len(loader.dataset)
    cpx = complexity_weighted_accuracy(seqs, true, pred)
    return avg_loss, cpx, pred, true, seqs


# ============ Hyper-parameter tuning loop ============
from torch_geometric.loader import DataLoader

batch_sizes = [32, 64, 128]
epochs = 5
experiment_data = {"batch_size": {"SPR": {}}}

for bs in batch_sizes:
    print(f"\n=== Training with batch size {bs} ===")
    # loaders
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=bs, shuffle=False)
    # fresh model & optimizer
    model = SPR_GNN(len(token2idx), n_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_cpx, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_cpx, _, _, _ = run_epoch(model, val_loader, criterion)
        run_record["losses"]["train"].append(tr_loss)
        run_record["losses"]["val"].append(v_loss)
        run_record["metrics"]["train"].append(tr_cpx)
        run_record["metrics"]["val"].append(v_cpx)
        run_record["epochs"].append(epoch)
        print(
            f"Epoch {epoch}: "
            f"train_loss={tr_loss:.4f}, val_loss={v_loss:.4f}, "
            f"train_CpxWA={tr_cpx:.4f}, val_CpxWA={v_cpx:.4f}, "
            f"elapsed={time.time()-t0:.1f}s"
        )
    # test evaluation
    te_loss, te_cpx, te_pred, te_true, _ = run_epoch(model, test_loader, criterion)
    print(f"Test : loss={te_loss:.4f}, CpxWA={te_cpx:.4f}")
    run_record["losses"]["test"] = te_loss
    run_record["metrics"]["test"] = te_cpx
    run_record["predictions"] = te_pred
    run_record["ground_truth"] = te_true
    experiment_data["batch_size"]["SPR"][f"bs{bs}"] = run_record

# ============ Save ============
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
