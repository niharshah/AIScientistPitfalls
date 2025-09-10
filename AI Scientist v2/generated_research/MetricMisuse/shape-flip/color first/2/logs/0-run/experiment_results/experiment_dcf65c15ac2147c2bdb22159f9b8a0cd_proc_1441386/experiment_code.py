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
    def _load(fname):  # helper
        return load_dataset(
            "csv", data_files=str(path / fname), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train.csv", "dev.csv", "test.csv"]:
        d[split.split(".")[0]] = _load(split)
    return d


def get_dataset():
    path_env = os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        ds = load_spr_bench(pathlib.Path(path_env))
        print("Loaded SPR_BENCH from", path_env)
    except Exception as e:
        print("Dataset not found, using synthetic set:", e)
        shapes, colors = "ABC", "XYZ"

        def rand_seq():
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

        ds = DatasetDict()
        ds["train"] = Dataset.from_dict(make_split(400))
        ds["dev"] = Dataset.from_dict(make_split(100))
        ds["test"] = Dataset.from_dict(make_split(100))
    return ds


dset = get_dataset()

# ---------- vocab ----------
all_tokens = set()
for split in dset.values():
    for seq in split["sequence"]:
        all_tokens.update(seq.split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id) + 1  # padding=0


# ---------- graph construction ----------
def seq_to_graph(sequence: str, label: int):
    toks = sequence.split()
    n = len(toks)
    x = torch.tensor([token2id[t] for t in toks], dtype=torch.long)
    edge = []
    for i in range(n - 1):
        edge.append([i, i + 1])
        edge.append([i + 1, i])
    edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
    return Data(
        x=x, edge_index=edge, y=torch.tensor([label], dtype=torch.long), seq=sequence
    )


def build_graph_list(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs, dev_graphs = build_graph_list(dset["train"]), build_graph_list(
    dset["dev"]
)
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)

num_classes = len(set(dset["train"]["label"]))


# ---------- model with dropout ----------
class GCN(nn.Module):
    def __init__(self, vocab, classes, dropout_rate: float):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.conv1, self.conv2 = GCNConv(64, 128), GCNConv(128, 128)
        self.dp = nn.Dropout(p=dropout_rate)
        self.lin = nn.Linear(128, classes)

    def forward(self, data):
        x = self.emb(data.x).to(device)
        x = torch.relu(self.conv1(x, data.edge_index))
        x = self.dp(x)
        x = torch.relu(self.conv2(x, data.edge_index))
        x = self.dp(x)
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ---------- experiment data ----------
experiment_data = {"dropout_rate_tuning": {"SPR_BENCH": {}}}

criterion = nn.CrossEntropyLoss()

# ---------- hyper-parameter sweep ----------
for rate in np.arange(0.0, 0.51, 0.1):
    rate = round(float(rate), 2)
    print(f"\n=== Training with dropout={rate} ===")
    model = GCN(vocab_size, num_classes, dropout_rate=rate).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    data_record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    epochs = 5
    for ep in range(1, epochs + 1):
        # ----- train -----
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch.num_graphs
        tr_loss /= len(train_graphs)

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        preds, labels, seqs = [], [], []
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

        ts = time.time()
        data_record["losses"]["train"].append((ts, tr_loss))
        data_record["losses"]["val"].append((ts, val_loss))
        data_record["metrics"]["train"].append(None)
        data_record["metrics"]["val"].append((ts, dwa))
        data_record["predictions"] = preds
        data_record["ground_truth"] = labels
        data_record["timestamps"].append(ts)

        print(
            f"Ep {ep} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | DWA {dwa:.4f}"
        )
    experiment_data["dropout_rate_tuning"]["SPR_BENCH"][str(rate)] = data_record

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
