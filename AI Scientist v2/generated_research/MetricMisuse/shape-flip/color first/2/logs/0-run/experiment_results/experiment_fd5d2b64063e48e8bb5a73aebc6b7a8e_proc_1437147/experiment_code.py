import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_color_variety(s) for s in sequences]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_shape_variety(s) for s in sequences]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def dual_weighted_accuracy(seqs, y_true, y_pred):
    return 0.5 * (
        color_weighted_accuracy(seqs, y_true, y_pred)
        + shape_weighted_accuracy(seqs, y_true, y_pred)
    )


# ---------- dataset loader ----------
def load_spr_bench(path: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(path / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _load(pathlib.Path("train.csv"))
    d["dev"] = _load(pathlib.Path("dev.csv"))
    d["test"] = _load(pathlib.Path("test.csv"))
    return d


def get_dataset():
    path_env = os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    path = pathlib.Path(path_env)
    try:
        dset = load_spr_bench(path)
        print("Loaded SPR_BENCH from", path)
    except Exception as e:
        # synthetic fallback
        print("Dataset not found, creating synthetic toy data:", e)

        def rand_seq():
            shapes = "ABC"
            colors = "XYZ"
            return " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(3, 8))
            )

        def make_split(n):
            return {
                "id": list(range(n)),
                "sequence": [rand_seq() for _ in range(n)],
                "label": [random.randint(0, 3) for _ in range(n)],
            }

        from datasets import Dataset

        dset = DatasetDict()
        dset["train"] = Dataset.from_dict(make_split(200))
        dset["dev"] = Dataset.from_dict(make_split(50))
        dset["test"] = Dataset.from_dict(make_split(50))
    return dset


dset = get_dataset()

# ---------- vocab ----------
all_tokens = set()
for split in dset.values():
    for seq in split["sequence"]:
        all_tokens.update(seq.split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id) + 1  # 0 for padding


# ---------- graph construction ----------
def seq_to_graph(sequence: str, label: int):
    tokens = sequence.split()
    n = len(tokens)
    x = torch.tensor([token2id[tok] for tok in tokens], dtype=torch.long)
    edge_index = []
    for i in range(n - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        seq=sequence,
    )


def build_graph_list(split):
    return [
        seq_to_graph(seq, lbl) for seq, lbl in zip(split["sequence"], split["label"])
    ]


train_graphs = build_graph_list(dset["train"])
dev_graphs = build_graph_list(dset["dev"])

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)


# ---------- model ----------
class GCN(nn.Module):
    def __init__(self, vocab, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.conv1 = GCNConv(64, 128)
        self.conv2 = GCNConv(128, 128)
        self.lin = nn.Linear(128, num_classes)

    def forward(self, data):
        x = self.emb(data.x).to(device)
        x = self.conv1(x, data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


num_classes = len(set(dset["train"]["label"]))
model = GCN(vocab_size, num_classes).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------- training loop ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    train_loss /= len(train_graphs)

    # validation
    model.eval()
    val_loss = 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            val_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1).cpu().tolist()
            labels = batch.y.view(-1).cpu().tolist()
            seqs = batch.seq
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(seqs)
    val_loss /= len(dev_graphs)
    dwa = dual_weighted_accuracy(all_seqs, all_labels, all_preds)

    # logging
    print(
        f"Epoch {epoch}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, DWA = {dwa:.4f}"
    )
    ts = time.time()
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ts, train_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ts, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ts, dwa))
    experiment_data["SPR_BENCH"]["predictions"] = all_preds
    experiment_data["SPR_BENCH"]["ground_truth"] = all_labels
    experiment_data["SPR_BENCH"]["timestamps"].append(ts)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
