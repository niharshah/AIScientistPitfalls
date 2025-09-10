import os, math, copy, random, time
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool, BatchNorm

# ------------------------------------------------------------
# Set up working directory and device
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# Helper functions for SPR symbols
# ------------------------------------------------------------
def colour_of(tok: str) -> str:
    return tok[1:] if len(tok) > 1 else ""


def shape_of(tok: str) -> str:
    return tok[0]


def count_colour_variety(seq: str) -> int:
    return len(set(colour_of(t) for t in seq.split()))


def count_shape_variety(seq: str) -> int:
    return len(set(shape_of(t) for t in seq.split()))


def cswa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    """Combined Structural Weighted Accuracy"""
    weights = [count_colour_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1, sum(weights))


def cwa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_colour_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1, sum(weights))


def swa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1, sum(weights))


# ------------------------------------------------------------
# Data loading – real data if present, otherwise small synthetic
# ------------------------------------------------------------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path: str) -> DatasetDict:
    if os.path.exists(os.path.join(path, "train.csv")):
        print("Loading SPR_BENCH from", path)

        def _ld(split: str):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))

    # ---------- tiny synthetic fallback ----------
    print("SPR_BENCH not found – generating toy data")
    shapes = list("ABCDEFGHI")
    colours = [str(i) for i in range(15)]

    def mk_seq() -> str:
        return " ".join(
            random.choice(shapes) + random.choice(colours)
            for _ in range(random.randint(5, 25))
        )

    def label_rule(seq: str) -> int:
        return sum(int(colour_of(t)) for t in seq.split()) % 2

    def mk_split(n: int) -> Dataset:
        seqs = [mk_seq() for _ in range(n)]
        labs = [label_rule(s) for s in seqs]
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labs}
        )

    return DatasetDict(train=mk_split(800), dev=mk_split(200), test=mk_split(200))


spr = load_spr(SPR_PATH)
print({k: len(v) for k, v in spr.items()})
num_classes = len(set(spr["train"]["label"]))

# ------------------------------------------------------------
# Build vocabularies & discover maximum length
# ------------------------------------------------------------
shape_vocab: Dict[str, int] = {}
colour_vocab: Dict[str, int] = {}
max_len = 0


def add_shape(s: str):
    if s not in shape_vocab:
        shape_vocab[s] = len(shape_vocab)


def add_colour(c: str):
    if c not in colour_vocab:
        colour_vocab[c] = len(colour_vocab)


for seq in spr["train"]["sequence"]:
    toks = seq.split()
    max_len = max(max_len, len(toks))
    for tok in toks:
        add_shape(shape_of(tok))
        add_colour(colour_of(tok))

print(
    "Vocab sizes – shapes:",
    len(shape_vocab),
    "colours:",
    len(colour_vocab),
    "| max_len:",
    max_len,
)


# ------------------------------------------------------------
# Sequence → graph encoding  (still ablating self-loops)
# ------------------------------------------------------------
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shape_ids = torch.tensor([shape_vocab[shape_of(t)] for t in toks], dtype=torch.long)
    colour_ids = torch.tensor(
        [colour_vocab[colour_of(t)] for t in toks], dtype=torch.long
    )
    pos_ids = torch.tensor(list(range(n)), dtype=torch.long)

    edges: List[List[int]] = [[i, i + 1] for i in range(n - 1)] + [
        [i + 1, i] for i in range(n - 1)
    ]
    for i in range(n):
        for j in range(i + 1, n):
            if shape_of(toks[i]) == shape_of(toks[j]):
                edges += [[i, j], [j, i]]
            if colour_of(toks[i]) == colour_of(toks[j]):
                edges += [[i, j], [j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(
        shape_id=shape_ids,
        colour_id=colour_ids,
        pos_id=pos_ids,
        edge_index=edge_index,
        y=torch.tensor(label, dtype=torch.long),
        seq=seq,
    )


def encode_split(ds: Dataset) -> List[Data]:
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, (spr["train"], spr["dev"], spr["test"])
)

# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------
batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ------------------------------------------------------------
# GNN model (positional embedding sized to true max_len)
# ------------------------------------------------------------
class GNNClassifier(nn.Module):
    def __init__(
        self, n_shape: int, n_col: int, pos_cap: int, hid: int = 64, n_classes: int = 2
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, hid)
        self.col_emb = nn.Embedding(n_col, hid)
        self.pos_emb = nn.Embedding(pos_cap + 1, hid)  # bug-fix: dynamic size
        self.conv1 = SAGEConv(hid, hid)
        self.bn1 = BatchNorm(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2 = BatchNorm(hid)
        self.out = nn.Linear(hid, n_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, data: Data):
        x = (
            self.shape_emb(data.shape_id)
            + self.col_emb(data.colour_id)
            + self.pos_emb(data.pos_id.clamp_max(self.pos_emb.num_embeddings - 1))
        )
        x = torch.relu(self.bn1(self.conv1(x, data.edge_index)))
        x = self.drop(x)
        x = torch.relu(self.bn2(self.conv2(x, data.edge_index)))
        x = global_add_pool(x, data.batch)
        return self.out(x)


model = GNNClassifier(len(shape_vocab), len(colour_vocab), max_len, 64, num_classes).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ------------------------------------------------------------
# Book-keeping dict
# ------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
        "best_epoch": None,
    }
}


# ------------------------------------------------------------
# Training / Evaluation helpers
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss = tot_correct = tot = 0
    seqs_all, preds_all, true_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        gts = batch.y.cpu().tolist()
        seqs = batch.seq
        tot_correct += sum(p == g for p, g in zip(preds, gts))
        tot += batch.num_graphs
        seqs_all.extend(seqs)
        preds_all.extend(preds)
        true_all.extend(gts)
    acc = tot_correct / tot
    cwa_ = cwa(seqs_all, true_all, preds_all)
    swa_ = swa(seqs_all, true_all, preds_all)
    cswa_ = cswa(seqs_all, true_all, preds_all)
    return tot_loss / tot, acc, cwa_, swa_, cswa_, seqs_all, preds_all, true_all


def train_epoch(loader):
    model.train()
    tot_loss = tot_correct = tot = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        gts = batch.y.cpu().tolist()
        tot_correct += sum(p == g for p, g in zip(preds, gts))
        tot += batch.num_graphs
    return tot_loss / tot, tot_correct / tot


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
max_epochs, patience = 40, 8
best_val_loss = math.inf
best_state = None
wait = 0

for epoch in range(1, max_epochs + 1):
    tr_loss, tr_acc = train_epoch(train_loader)
    val_loss, val_acc, val_cwa, val_swa, val_cswa, *_ = evaluate(dev_loader)

    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"epoch": epoch, "acc": tr_acc})
    ed["metrics"]["val"].append(
        {
            "epoch": epoch,
            "acc": val_acc,
            "CWA": val_cwa,
            "SWA": val_swa,
            "CSWA": val_cswa,
        }
    )
    print(
        f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  acc={val_acc:.3f} "
        f"CWA={val_cwa:.3f} SWA={val_swa:.3f} CSWA={val_cswa:.3f}"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        ed["best_epoch"] = epoch
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ------------------------------------------------------------
# Test set evaluation
# ------------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

test_loss, test_acc, test_cwa, test_swa, test_cswa, seqs, preds, gts = evaluate(
    test_loader
)
print(
    f"TEST -- loss:{test_loss:.4f} acc:{test_acc:.3f} "
    f"CWA:{test_cwa:.3f} SWA:{test_swa:.3f} CSWA:{test_cswa:.3f}"
)

ed = experiment_data["SPR_BENCH"]
ed["predictions"] = preds
ed["ground_truth"] = gts
ed["sequences"] = seqs

# ------------------------------------------------------------
# Persist results
# ------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
