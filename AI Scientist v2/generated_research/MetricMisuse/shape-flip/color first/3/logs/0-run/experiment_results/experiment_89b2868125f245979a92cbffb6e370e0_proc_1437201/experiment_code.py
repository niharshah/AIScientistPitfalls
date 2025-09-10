import os, random, string, itertools, time, pathlib, numpy as np, torch
from typing import List
from torch import nn
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers ----------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def balanced_weighted_accuracy(seqs, y_true, y_pred):
    return 0.5 * (
        color_weighted_accuracy(seqs, y_true, y_pred)
        + shape_weighted_accuracy(seqs, y_true, y_pred)
    )


# ---------- dataset loading or synthesis ----------
def create_synthetic_csv(path: pathlib.Path, n_rows: int):
    shapes = list(string.ascii_uppercase[:4])  # A,B,C,D
    colors = list("1234")
    with open(path, "w") as f:
        f.write("id,sequence,label\n")
        for idx in range(n_rows):
            L = random.randint(4, 10)
            toks = [
                "".join(random.choices(shapes, k=1) + random.choices(colors, k=1))
                for _ in range(L)
            ]
            seq = " ".join(toks)
            # simple hidden rule: label 1 if number of unique shapes is even else 0
            label = int(len(set(t[0] for t in toks)) % 2 == 0)
            f.write(f"{idx},{seq},{label}\n")


def ensure_dataset():
    root = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    if not root.exists():
        print("SPR_BENCH not found; generating synthetic data.")
        root.mkdir(exist_ok=True)
        create_synthetic_csv(root / "train.csv", 2000)
        create_synthetic_csv(root / "dev.csv", 500)
        create_synthetic_csv(root / "test.csv", 500)
    return root


DATA_PATH = ensure_dataset()


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


spr_bench = load_spr_bench(DATA_PATH)

# ---------- Graph construction ----------
shape_vocab = {c: i for i, c in enumerate(string.ascii_uppercase)}
color_vocab = {c: i for i, c in enumerate(string.digits)}


def seq_to_graph(seq: str, label: int, idx: int) -> Data:
    toks = seq.strip().split()
    n = len(toks)
    shape_idx = torch.tensor([shape_vocab[t[0]] for t in toks], dtype=torch.long)
    color_idx = torch.tensor([color_vocab[t[1]] for t in toks], dtype=torch.long)
    # edges: connect i<->i+1
    if n > 1:
        edge_index = (
            torch.tensor(
                [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(
        shape=shape_idx,
        color=color_idx,
        edge_index=edge_index,
        num_nodes=n,
        y=y,
        seq=seq,
        idx=idx,
    )


class SPRGraphDataset(InMemoryDataset):
    def __init__(self, hf_split, transform=None):
        super().__init__(".", transform)
        data_list = []
        for ex in hf_split:
            data_list.append(
                seq_to_graph(ex["sequence"], int(ex["label"]), int(ex["id"]))
            )
        self.data, self.slices = self.collate(data_list)


train_dataset = SPRGraphDataset(spr_bench["train"])
dev_dataset = SPRGraphDataset(spr_bench["dev"])
test_dataset = SPRGraphDataset(spr_bench["test"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)


# ---------- Model ----------
class SPRGCN(nn.Module):
    def __init__(self, shape_dim=8, color_dim=4, hidden=32, num_classes=2):
        super().__init__()
        self.shape_emb = nn.Embedding(26, shape_dim)
        self.color_emb = nn.Embedding(10, color_dim)
        in_dim = shape_dim + color_dim
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = torch.cat([self.shape_emb(data.shape), self.color_emb(data.color)], dim=1)
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)


model = SPRGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- training loop ----------
def evaluate(loader):
    model.eval()
    ys, preds, seqs = [], [], []
    loss_accum, n = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            loss_accum += loss.item() * data.y.size(0)
            n += data.y.size(0)
            ys.extend(data.y.cpu().tolist())
            preds.extend(out.argmax(1).cpu().tolist())
            seqs.extend(data.seq)
    avg_loss = loss_accum / n
    bwa = balanced_weighted_accuracy(seqs, ys, preds)
    return avg_loss, bwa, ys, preds, seqs


EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total = 0.0, 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
        total += data.y.size(0)
    train_loss = total_loss / total

    val_loss, val_bwa, _, _, _ = evaluate(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_bwa)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, validation_loss = {val_loss:.4f}, val_BWA={val_bwa:.4f}"
    )

# ---------- final evaluation ----------
test_loss, test_bwa, ys, preds, seqs = evaluate(test_loader)
print(f"Test   : loss={test_loss:.4f}, BWA={test_bwa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = ys
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
