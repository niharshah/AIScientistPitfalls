import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ---------- experiment dict ----------
experiment_data = {"batch_size_tuning": {}}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- metrics ----------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def dual_weighted_accuracy(seqs, y_t, y_p):
    return 0.5 * (
        color_weighted_accuracy(seqs, y_t, y_p)
        + shape_weighted_accuracy(seqs, y_t, y_p)
    )


# ---------- dataset loader ----------
def load_spr_bench(path: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(path / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def get_dataset():
    path_env = os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        d = load_spr_bench(pathlib.Path(path_env))
        print("Loaded SPR_BENCH from", path_env)
    except Exception as e:
        print("Dataset not found, generating synthetic:", e)

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

        d = DatasetDict(
            train=Dataset.from_dict(make_split(200)),
            dev=Dataset.from_dict(make_split(50)),
            test=Dataset.from_dict(make_split(50)),
        )
    return d


dset = get_dataset()

# ---------- vocab ----------
all_tokens = set(
    tok
    for seq in dset["train"]["sequence"]
    + dset["dev"]["sequence"]
    + dset["test"]["sequence"]
    for tok in seq.split()
)
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id) + 1  # 0 padding

# ---------- graph helpers ----------
from torch_geometric.data import Data


def seq_to_graph(sequence: str, label: int):
    tokens = sequence.split()
    n = len(tokens)
    x = torch.tensor([token2id[t] for t in tokens], dtype=torch.long)
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
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs = build_graph_list(dset["train"])
dev_graphs = build_graph_list(dset["dev"])


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
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


num_classes = len(set(dset["train"]["label"]))
criterion = nn.CrossEntropyLoss()

# ---------- hyperparameter sweep ----------
batch_sizes = [16, 32, 64, 128]
experiment_data["batch_size_tuning"]["SPR_BENCH"] = {}
epochs = 5
for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    # loaders
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=max(bs * 2, 128), shuffle=False)
    # model & optimizer
    model = GCN(vocab_size, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    # storage
    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_graphs)
        # val
        model.eval()
        val_loss = 0.0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.view(-1))
                val_loss += loss.item() * batch.num_graphs
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(batch.y.view(-1).cpu().tolist())
                seqs.extend(batch.seq)
        val_loss /= len(dev_graphs)
        dwa = dual_weighted_accuracy(seqs, labels, preds)
        # log
        ts = time.time()
        exp_entry["losses"]["train"].append((ts, train_loss))
        exp_entry["losses"]["val"].append((ts, val_loss))
        exp_entry["metrics"]["train"].append(None)
        exp_entry["metrics"]["val"].append((ts, dwa))
        exp_entry["predictions"] = preds
        exp_entry["ground_truth"] = labels
        exp_entry["timestamps"].append(ts)
        print(
            f"Epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | DWA {dwa:.4f}"
        )
    experiment_data["batch_size_tuning"]["SPR_BENCH"][str(bs)] = exp_entry

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
