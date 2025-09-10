# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# hyperparameter-tuning : pooling_type
import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    GlobalAttention,
)
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- metrics ----------
def count_color_variety(sequence):
    return len(set(t[1] for t in sequence.split() if len(t) > 1))


def count_shape_variety(sequence):
    return len(set(t[0] for t in sequence.split() if t))


def color_weighted_accuracy(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def dual_weighted_accuracy(seqs, y, p):
    return 0.5 * (
        color_weighted_accuracy(seqs, y, p) + shape_weighted_accuracy(seqs, y, p)
    )


# ---------- dataset loader ----------
def load_spr_bench(path):
    def _l(name):
        return load_dataset(
            "csv", data_files=str(path / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _l(pathlib.Path("train.csv"))
    d["dev"] = _l(pathlib.Path("dev.csv"))
    d["test"] = _l(pathlib.Path("test.csv"))
    return d


def get_dataset():
    path_env = os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        d = load_spr_bench(pathlib.Path(path_env))
        print("Loaded SPR_BENCH from", path_env)
    except Exception as e:
        print("Dataset not found, creating synthetic data:", e)
        shapes, colors = "ABC", "XYZ"

        def rand_seq():
            return " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(3, 8))
            )

        def make(n):
            return {
                "id": list(range(n)),
                "sequence": [rand_seq() for _ in range(n)],
                "label": [random.randint(0, 3) for _ in range(n)],
            }

        from datasets import Dataset

        d = DatasetDict()
        d["train"] = Dataset.from_dict(make(200))
        d["dev"] = Dataset.from_dict(make(50))
        d["test"] = Dataset.from_dict(make(50))
    return d


dset = get_dataset()

# ---------- vocab ----------
all_tokens = set(
    t for split in dset.values() for seq in split["sequence"] for t in seq.split()
)
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id) + 1

# ---------- graphs ----------
from torch_geometric.data import Data


def seq_to_graph(seq, lbl):
    toks = seq.split()
    n = len(toks)
    x = torch.tensor([token2id[t] for t in toks], dtype=torch.long)
    edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([lbl], dtype=torch.long), seq=seq
    )


def build(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs, dev_graphs = build(dset["train"]), build(dset["dev"])
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)


# ---------- model ----------
class GCN(nn.Module):
    def __init__(self, vocab, num_classes, pooling="mean"):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.conv1, self.conv2 = GCNConv(64, 128), GCNConv(128, 128)
        self.pooling_type = pooling
        if pooling == "mean":
            self.pool = lambda x, b: global_mean_pool(x, b)
        elif pooling == "max":
            self.pool = lambda x, b: global_max_pool(x, b)
        elif pooling == "add":
            self.pool = lambda x, b: global_add_pool(x, b)
        elif pooling == "attn":
            gate = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.attn = GlobalAttention(gate)
            self.pool = lambda x, b: self.attn(x, b)
        self.lin = nn.Linear(128, num_classes)

    def forward(self, data):
        x = self.emb(data.x).to(device)
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = self.pool(x, data.batch)
        return self.lin(x)


num_classes = len(set(dset["train"]["label"]))
pool_options = ["mean", "max", "add", "attn"]
experiment_data = {"pooling_type": {}}
epochs = 5

for pool in pool_options:
    print(f"\n=== Training with {pool} pooling ===")
    model = GCN(vocab_size, num_classes, pool).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tloss = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y.view(-1))
            loss.backward()
            opt.step()
            tloss += loss.item() * batch.num_graphs
        tloss /= len(train_graphs)
        # val
        model.eval()
        vloss = 0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = crit(out, batch.y.view(-1))
                vloss += loss.item() * batch.num_graphs
                preds += out.argmax(1).cpu().tolist()
                labels += batch.y.view(-1).cpu().tolist()
                seqs += batch.seq
        vloss /= len(dev_graphs)
        dwa = dual_weighted_accuracy(seqs, labels, preds)
        ts = time.time()
        log["losses"]["train"].append((ts, tloss))
        log["losses"]["val"].append((ts, vloss))
        log["metrics"]["train"].append(None)
        log["metrics"]["val"].append((ts, dwa))
        log["predictions"], log["ground_truth"] = preds, labels
        log["timestamps"].append(ts)
        print(
            f"Epoch {ep}/{epochs} | TrainLoss {tloss:.4f} | ValLoss {vloss:.4f} | DWA {dwa:.4f}"
        )
    experiment_data["pooling_type"][pool] = {"SPR_BENCH": log}

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved results to", os.path.join(working_dir, "experiment_data.npy"))
