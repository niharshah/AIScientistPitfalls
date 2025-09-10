import os, pathlib, time, numpy as np, torch, torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from datasets import load_dataset, DatasetDict
from typing import Dict, List

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- GPU / CPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment container ----------
experiment_data = {
    "spr_bench": {
        "metrics": {"train_BWA": [], "val_BWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ---------- locate dataset ----------
def locate_spr_bench() -> pathlib.Path:
    """Find SPR_BENCH directory via env var or upward search."""
    env_dir = os.getenv("SPR_BENCH_DIR")
    if env_dir and pathlib.Path(env_dir).is_dir():
        return pathlib.Path(env_dir)
    cwd = pathlib.Path.cwd()
    for p in [cwd] + list(cwd.parents):
        cand = p / "SPR_BENCH"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Unable to locate SPR_BENCH directory. "
        "Set environment variable SPR_BENCH_DIR or place 'SPR_BENCH' folder in the project tree."
    )


# ---------- dataset utilities ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs: List[str], y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def shape_weighted_accuracy(seqs: List[str], y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# ---------- graph construction ----------
token2id: Dict[str, int] = {}


def seq_to_graph(sequence: str, label: int) -> Data:
    """Convert token sequence to a simple chain graph."""
    global token2id
    toks = sequence.strip().split()
    node_ids = []
    for tok in toks:
        if tok not in token2id:
            token2id[tok] = len(token2id)
        node_ids.append(token2id[tok])

    x = torch.tensor(node_ids, dtype=torch.long)  # node indices
    if len(toks) > 1:
        src = list(range(len(toks) - 1)) + list(range(1, len(toks)))
        dst = list(range(1, len(toks))) + list(range(len(toks) - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        seq=sequence,
    )


# ---------- load data ----------
DATA_PATH = locate_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")
spr = load_spr_bench(DATA_PATH)

# label mapping (ensure contiguous 0…C-1)
label_values = sorted(
    {int(lbl) for split in ["train", "dev", "test"] for lbl in spr[split]["label"]}
)
label_map = {v: i for i, v in enumerate(label_values)}
num_classes = len(label_values)


def build_graphs(split: str) -> List[Data]:
    return [
        seq_to_graph(seq, label_map[int(lbl)])
        for seq, lbl in zip(spr[split]["sequence"], spr[split]["label"])
    ]


train_graphs = build_graphs("train")
dev_graphs = build_graphs("dev")
test_graphs = build_graphs("test")


# ---------- model ----------
class GNNClassifier(nn.Module):
    def __init__(self, num_tokens: int, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, batch):
        x = self.embed(batch.x)  # (num_nodes, hidden)
        x = F.relu(self.conv1(x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = global_mean_pool(x, batch.batch)  # (num_graphs, hidden)
        return self.lin(x)


model = GNNClassifier(num_tokens=len(token2id), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- data loaders ----------
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)

# ---------- training loop ----------
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    # ---- training ----
    model.train()
    running_loss = 0.0
    train_y_true, train_y_pred, train_seqs = [], [], []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1).cpu().tolist()
        train_y_pred.extend(preds)
        train_y_true.extend(batch.y.view(-1).cpu().tolist())
        train_seqs.extend(batch.seq)

    train_loss = running_loss / len(train_loader.dataset)
    cwa_train = color_weighted_accuracy(train_seqs, train_y_true, train_y_pred)
    swa_train = shape_weighted_accuracy(train_seqs, train_y_true, train_y_pred)
    bwa_train = (cwa_train + swa_train) / 2.0

    # ---- validation ----
    model.eval()
    val_loss_tot = 0.0
    val_y_true, val_y_pred, val_seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            val_loss_tot += criterion(out, batch.y.view(-1)).item() * batch.num_graphs
            preds = out.argmax(dim=1).cpu().tolist()
            val_y_pred.extend(preds)
            val_y_true.extend(batch.y.view(-1).cpu().tolist())
            val_seqs.extend(batch.seq)

    val_loss = val_loss_tot / len(dev_loader.dataset)
    cwa_val = color_weighted_accuracy(val_seqs, val_y_true, val_y_pred)
    swa_val = shape_weighted_accuracy(val_seqs, val_y_true, val_y_pred)
    bwa_val = (cwa_val + swa_val) / 2.0

    # ---- logging ----
    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_BWA={bwa_train:.4f} val_BWA={bwa_val:.4f}"
    )

    experiment_data["spr_bench"]["losses"]["train"].append(train_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)
    experiment_data["spr_bench"]["metrics"]["train_BWA"].append(bwa_train)
    experiment_data["spr_bench"]["metrics"]["val_BWA"].append(bwa_val)
    experiment_data["spr_bench"]["timestamps"].append(time.time())

# ---------- final test evaluation ----------
model.eval()
test_y_true, test_y_pred, test_seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds = out.argmax(dim=1).cpu().tolist()
        test_y_pred.extend(preds)
        test_y_true.extend(batch.y.view(-1).cpu().tolist())
        test_seqs.extend(batch.seq)

cwa_test = color_weighted_accuracy(test_seqs, test_y_true, test_y_pred)
swa_test = shape_weighted_accuracy(test_seqs, test_y_true, test_y_pred)
bwa_test = (cwa_test + swa_test) / 2.0

print(f"\nTest Results -> CWA={cwa_test:.4f} SWA={swa_test:.4f} BWA={bwa_test:.4f}")

experiment_data["spr_bench"]["predictions"] = test_y_pred
experiment_data["spr_bench"]["ground_truth"] = test_y_true
experiment_data["spr_bench"]["metrics"]["test_BWA"] = bwa_test

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
