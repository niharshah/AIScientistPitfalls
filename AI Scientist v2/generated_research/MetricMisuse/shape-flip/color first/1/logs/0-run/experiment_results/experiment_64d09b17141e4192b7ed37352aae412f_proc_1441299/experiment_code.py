import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------------- working dir & device ------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# --------------------------------------------------------------


# ---------------- helper: metrics -----------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def complexity_adjusted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# --------------------------------------------------------------


# --------------- data loading / generation --------------------
def try_load_benchmark():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    dset = DatasetDict()
    dset["train"], dset["dev"], dset["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return dset


def generate_synthetic(ntr=200, ndev=60, ntst=100):
    shapes, colors = list(string.ascii_uppercase[:6]), list(string.ascii_lowercase[:6])

    def _gen(n):
        seqs, labels = [], []
        for _ in range(n):
            tokens = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 15))
            ]
            seqs.append(" ".join(tokens))
            labels.append(int(tokens[0][0] == tokens[-1][0]))
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(_gen(ntr)),
            "dev": Dataset.from_dict(_gen(ndev)),
            "test": Dataset.from_dict(_gen(ntst)),
        }
    )


dataset = try_load_benchmark()
if dataset is None:
    print("Benchmark not found; generating synthetic.")
    dataset = generate_synthetic()
num_classes = len(set(dataset["train"]["label"]))
print(
    f"Dataset: {len(dataset['train'])} train, {len(dataset['dev'])} dev, classes={num_classes}"
)
# --------------------------------------------------------------

# ---------------- vocabulary build ----------------------------
vocab = {}


def add_token(t):
    if t not in vocab:
        vocab[t] = len(vocab)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)
# --------------------------------------------------------------


# ---------------- graph construction --------------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    node_ids = torch.tensor([vocab[t] for t in toks], dtype=torch.long)
    if len(toks) > 1:
        src = list(range(len(toks) - 1))
        dst = list(range(1, len(toks)))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    d = Data(
        x=node_ids.unsqueeze(-1),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )
    d.seq_raw = seq
    return d


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
# --------------------------------------------------------------


# ----------------- model definition ---------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab, embed_dim=32, hidden_dim=64, num_classes=2, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = self.drop(F.relu(self.conv1(x, data.edge_index)))
        x = self.drop(F.relu(self.conv2(x, data.edge_index)))
        x = global_mean_pool(x, data.batch)
        x = self.drop(x)
        return self.lin(x)


# --------------------------------------------------------------

# ------------------ experiment container ----------------------
experiment_data = {"dropout_tuning": {}}  # will fill with each rate
# --------------------------------------------------------------

dropout_rates = [0.0, 0.1, 0.25, 0.4, 0.5]
epochs = 5
for dr in dropout_rates:
    print(f"\n=== Training with dropout={dr} ===")
    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    model = GNNClassifier(vocab, num_classes=num_classes, dropout=dr).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=64)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_loader.dataset)
        exp_rec["losses"]["train"].append(train_loss)

        model.eval()
        dev_loss = 0.0
        preds, gts, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                dev_loss += loss.item() * batch.num_graphs
                preds.extend(logits.argmax(-1).cpu().tolist())
                gts.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq_raw)
        dev_loss /= len(dev_loader.dataset)
        exp_rec["losses"]["val"].append(dev_loss)

        acc = np.mean([p == t for p, t in zip(preds, gts)])
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        caa = complexity_adjusted_accuracy(seqs, gts, preds)
        exp_rec["metrics"]["val"].append(
            {"acc": acc, "cwa": cwa, "swa": swa, "caa": caa}
        )
        exp_rec["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={dev_loss:.4f}, "
            f"acc={acc:.3f}, CWA={cwa:.3f}, SWA={swa:.3f}, CAA={caa:.3f}"
        )

    exp_rec["predictions"] = preds
    exp_rec["ground_truth"] = gts
    experiment_data["dropout_tuning"][f"dropout_{dr}"] = exp_rec

# --------------- save experiment data -------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "All experiments complete. Saved to",
    os.path.join(working_dir, "experiment_data.npy"),
)
