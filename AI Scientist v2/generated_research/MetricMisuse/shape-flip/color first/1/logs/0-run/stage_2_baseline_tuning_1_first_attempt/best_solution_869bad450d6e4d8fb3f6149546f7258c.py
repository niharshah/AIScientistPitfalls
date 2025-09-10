import os, pathlib, random, string, time, math, json, warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ----------------- working directory & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------- helper: metrics ------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def complexity_adjusted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# --------------------- data loading ---------------------------
def try_load_benchmark():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

    def _load(csv_file):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"], dset["dev"], dset["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return dset


def generate_synthetic(num_train=200, num_dev=60, num_test=100):
    shapes = list(string.ascii_uppercase[:6])
    colors = list(string.ascii_lowercase[:6])

    def _gen(n):
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(4, 15)
            tokens = [
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            ]
            seq = " ".join(tokens)
            label = int(tokens[0][0] == tokens[-1][0])
            seqs.append(seq)
            labels.append(label)
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(_gen(num_train)),
            "dev": Dataset.from_dict(_gen(num_dev)),
            "test": Dataset.from_dict(_gen(num_test)),
        }
    )


dataset = try_load_benchmark()
if dataset is None:
    print("Benchmark not found; generating synthetic data.")
    dataset = generate_synthetic()

num_classes = len(set(dataset["train"]["label"]))
print(
    f"Loaded dataset with {len(dataset['train'])} train, {len(dataset['dev'])} dev samples, classes={num_classes}"
)

# ------------------ vocabulary build --------------------------
vocab = {}


def add_token(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")


# -------------------- graph construction ----------------------
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


# -------------------------- model -----------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab, embed_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ------------------ experiment tracking -----------------------
experiment_data = {"learning_rate": {"SPR": {}}}

# ------------------ hyperparameter sweep ----------------------
learning_rates = [3e-4, 1e-3, 3e-3]
epochs = 5
batch_size_train = 32
batch_size_eval = 64

for lr in learning_rates:
    lr_key = f"{lr:.0e}" if lr < 1 else str(lr)
    print(f"\n=== Training with learning rate {lr_key} ===")
    model = GNNClassifier(vocab, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_graphs, batch_size=batch_size_train, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=batch_size_eval)

    run_log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    for epoch in range(1, epochs + 1):
        # ---- train
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
        run_log["losses"]["train"].append(train_loss)

        # ---- validation
        model.eval()
        dev_loss, preds, gts, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                dev_loss += loss.item() * batch.num_graphs
                pred = logits.argmax(dim=-1).cpu().tolist()
                gt = batch.y.cpu().tolist()
                seqs.extend(batch.seq_raw)
                preds.extend(pred)
                gts.extend(gt)
        dev_loss /= len(dev_loader.dataset)
        run_log["losses"]["val"].append(dev_loss)

        acc = float(np.mean([p == t for p, t in zip(preds, gts)]))
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        caa = complexity_adjusted_accuracy(seqs, gts, preds)
        run_log["metrics"]["val"].append(
            {"acc": acc, "cwa": cwa, "swa": swa, "caa": caa}
        )
        run_log["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={dev_loss:.4f}, "
            f"acc={acc:.3f}, CWA={cwa:.3f}, SWA={swa:.3f}, CAA={caa:.3f}"
        )

    run_log["predictions"] = preds
    run_log["ground_truth"] = gts
    experiment_data["learning_rate"]["SPR"][lr_key] = run_log

# --------------- save experiment data -------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "All runs finished. Data saved to", os.path.join(working_dir, "experiment_data.npy")
)
