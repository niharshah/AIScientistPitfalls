import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# -------------------------------------------------------------- #
# experiment bookkeeping                                         #
# -------------------------------------------------------------- #
experiment_data = {
    "embed_dim": {  # hyperparameter being tuned
        "SPR": []  # list holding results for every embed_dim tried
    }
}

# -------------------------------------------------------------- #
# device & working dir                                           #
# -------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------- helper: metrics ------------------------- #
def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def complexity_adjusted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ----------------------- data utils --------------------------- #
def try_load_benchmark():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

    def _l(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    d = lambda x: _l(f"{x}.csv")
    return DatasetDict(train=d("train"), dev=d("dev"), test=d("test"))


def generate_synthetic(n_tr=200, n_dev=60, n_test=100):
    shapes = list(string.ascii_uppercase[:6])
    colors = list(string.ascii_lowercase[:6])

    def _gen(n):
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(4, 15)
            toks = [
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            ]
            seqs.append(" ".join(toks))
            labels.append(int(toks[0][0] == toks[-1][0]))
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        train=Dataset.from_dict(_gen(n_tr)),
        dev=Dataset.from_dict(_gen(n_dev)),
        test=Dataset.from_dict(_gen(n_test)),
    )


dataset = try_load_benchmark()
if dataset is None:
    print("Benchmark not found, generating synthetic dataset.")
    dataset = generate_synthetic()

num_classes = len(set(dataset["train"]["label"]))
print(
    f"Dataset sizes: train={len(dataset['train'])}, dev={len(dataset['dev'])}, test={len(dataset['test'])}"
)

# -------------------- build vocabulary ------------------------ #
vocab = {}


def add_token(t):
    vocab.setdefault(t, len(vocab))


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ------------------ sequence to graph ------------------------- #
def seq_to_graph(seq, label):
    tokens = seq.split()
    node_ids = torch.tensor([vocab[t] for t in tokens], dtype=torch.long)
    if len(tokens) > 1:
        src = list(range(len(tokens) - 1))
        dst = list(range(1, len(tokens)))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    data = Data(
        x=node_ids.unsqueeze(-1),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )
    data.seq_raw = seq
    return data


train_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["train"]["sequence"], dataset["train"]["label"])
]
dev_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["dev"]["sequence"], dataset["dev"]["label"])
]
test_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["test"]["sequence"], dataset["test"]["label"])
]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=64)


# -------------------------- model ----------------------------- #
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ------------------- training routine ------------------------- #
def run_experiment(embed_dim, epochs=5):
    model = GNNClassifier(
        vocab_size, embed_dim, hidden_dim=64, num_classes=num_classes
    ).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    metrics_train, metrics_val = [], []
    losses_train, losses_val = [], []
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tot = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y)
            loss.backward()
            optim.step()
            tot += loss.item() * batch.num_graphs
        tr_loss = tot / len(train_loader.dataset)
        losses_train.append(tr_loss)
        # val
        model.eval()
        tot = 0.0
        preds, gts, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = crit(logits, batch.y)
                tot += loss.item() * batch.num_graphs
                preds.extend(logits.argmax(-1).cpu().tolist())
                gts.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq_raw)
        val_loss = tot / len(dev_loader.dataset)
        losses_val.append(val_loss)
        acc = float(np.mean([p == t for p, t in zip(preds, gts)]))
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        caa = complexity_adjusted_accuracy(seqs, gts, preds)
        metrics_val.append({"acc": acc, "cwa": cwa, "swa": swa, "caa": caa})
        print(
            f"[embed_dim={embed_dim}] Epoch {ep}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, acc={acc:.3f}"
        )
    return {
        "embed_dim": embed_dim,
        "metrics": {"train": metrics_train, "val": metrics_val},
        "losses": {"train": losses_train, "val": losses_val},
        "predictions": preds,
        "ground_truth": gts,
        "timestamps": time.time(),
    }


# ----------------- hyperparameter sweep ----------------------- #
for dim in [16, 32, 64, 128]:
    result = run_experiment(dim, epochs=5)
    experiment_data["embed_dim"]["SPR"].append(result)

# ----------------------- save results ------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "All experiments complete. Data saved to",
    os.path.join(working_dir, "experiment_data.npy"),
)
