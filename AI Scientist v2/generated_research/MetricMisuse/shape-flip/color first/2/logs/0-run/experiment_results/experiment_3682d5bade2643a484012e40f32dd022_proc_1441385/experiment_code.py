import os, pathlib, random, time, numpy as np, torch
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
        print("Dataset not found, creating synthetic toy data:", e)

        def rand_seq():
            shapes, colors = "ABC", "XYZ"
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
    x = torch.tensor([token2id[t] for t in tokens], dtype=torch.long)
    edge_index = []
    for i in range(len(tokens) - 1):
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
    def __init__(self, vocab, num_classes, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim)
        self.conv1 = GCNConv(emb_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.lin = nn.Linear(128, num_classes)

    def forward(self, data):
        x = self.emb(data.x).to(device)
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ---------- experiment data container ----------
experiment_data = {"embedding_dim": {}}

num_classes = len(set(dset["train"]["label"]))
embedding_dims = [32, 64, 128]  # hyper-parameter sweep
epochs = 5
for emb_dim in embedding_dims:
    print(f"\n======== Training with embedding_dim = {emb_dim} ========")
    # dataloaders (recreated each sweep for shuffling independence)
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)

    model = GCN(vocab_size, num_classes, emb_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    key = f"dim_{emb_dim}"
    experiment_data["embedding_dim"].setdefault("SPR_BENCH", {})[key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    store = experiment_data["embedding_dim"]["SPR_BENCH"][key]

    for epoch in range(1, epochs + 1):
        # --- train ---
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

        # --- dev ---
        model.eval()
        val_loss, preds, labels, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss += criterion(out, batch.y.view(-1)).item() * batch.num_graphs
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(batch.y.view(-1).cpu().tolist())
                seqs.extend(batch.seq)
        val_loss /= len(dev_graphs)
        dwa = dual_weighted_accuracy(seqs, labels, preds)

        ts = time.time()
        store["losses"]["train"].append((ts, train_loss))
        store["losses"]["val"].append((ts, val_loss))
        store["metrics"]["train"].append(None)
        store["metrics"]["val"].append((ts, dwa))
        store["predictions"] = preds
        store["ground_truth"] = labels
        store["timestamps"].append(ts)
        print(
            f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | DWA {dwa:.4f}"
        )

# ---------- save all ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
