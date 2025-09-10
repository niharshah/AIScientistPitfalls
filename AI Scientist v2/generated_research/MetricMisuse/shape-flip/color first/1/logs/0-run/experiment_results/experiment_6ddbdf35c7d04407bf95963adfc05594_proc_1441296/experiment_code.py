import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# --------------- experiment bookkeeping -----------------
experiment_data = {
    "epochs": {  # hyper-parameter we tune
        "SPR": {  # dataset name
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
# --------------------------------------------------------

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------ metric helpers ----------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def _weighted_acc(weights, y_true, y_pred):
    return sum(w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)) / max(
        sum(weights), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    return _weighted_acc([count_color_variety(s) for s in seqs], y_true, y_pred)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    return _weighted_acc([count_shape_variety(s) for s in seqs], y_true, y_pred)


def complexity_adjusted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return _weighted_acc(w, y_true, y_pred)


# --------------------------------------------------------


# ----------------- dataset loading ----------------------
def try_load_benchmark():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def generate_synthetic(n_train=200, n_dev=60, n_test=100):
    shapes, colors = list(string.ascii_uppercase[:6]), list(string.ascii_lowercase[:6])

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
        train=Dataset.from_dict(_gen(n_train)),
        dev=Dataset.from_dict(_gen(n_dev)),
        test=Dataset.from_dict(_gen(n_test)),
    )


dataset = try_load_benchmark() or generate_synthetic()
num_classes = len(set(dataset["train"]["label"]))
print(
    f"Data: {len(dataset['train'])} train / {len(dataset['dev'])} dev samples, classes={num_classes}"
)

# ------------------ vocabulary --------------------------
vocab = {}


def add(tok):
    vocab.setdefault(tok, len(vocab))


for seq in dataset["train"]["sequence"]:
    for t in seq.split():
        add(t)
print("Vocabulary size:", len(vocab))


# --------------- graph construction ---------------------
def seq_to_graph(seq, label):
    node_ids = torch.tensor([vocab[t] for t in seq.split()], dtype=torch.long)
    if len(node_ids) > 1:
        src = list(range(len(node_ids) - 1))
        dst = list(range(1, len(node_ids)))
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

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=64)
test_loader = DataLoader(test_graphs, batch_size=64)


# ------------------- model ------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim=32, hidden_dim=64, n_cls=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, n_cls)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = GNNClassifier(len(vocab), n_cls=num_classes).to(device)
optimizer, criterion = Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss()

# -------------- training with more epochs ---------------
max_epochs, patience = 30, 5
best_val_loss, epochs_no_improve = float("inf"), 0

for epoch in range(1, max_epochs + 1):
    # ---- train ----
    model.train()
    tot_loss = 0
    correct = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        correct += (out.argmax(1) == batch.y).sum().item()
    train_loss = tot_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    experiment_data["epochs"]["SPR"]["losses"]["train"].append(train_loss)
    experiment_data["epochs"]["SPR"]["metrics"]["train"].append({"acc": train_acc})

    # ---- validation ----
    model.eval()
    v_loss, preds, gts, seqs = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            logits = model(batch)
            v_loss += criterion(logits, batch.y).item() * batch.num_graphs
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch.y.cpu().tolist())
            seqs.extend(batch.seq_raw)
    v_loss /= len(dev_loader.dataset)
    acc = np.mean([p == t for p, t in zip(preds, gts)])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    caa = complexity_adjusted_accuracy(seqs, gts, preds)
    experiment_data["epochs"]["SPR"]["losses"]["val"].append(v_loss)
    experiment_data["epochs"]["SPR"]["metrics"]["val"].append(
        {"acc": acc, "cwa": cwa, "swa": swa, "caa": caa}
    )
    experiment_data["epochs"]["SPR"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}/{max_epochs} "
        f"train_loss={train_loss:.4f} val_loss={v_loss:.4f} "
        f"val_acc={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CAA={caa:.3f}"
    )

    # early stopping
    if v_loss < best_val_loss - 1e-4:
        best_val_loss, epochs_no_improve = v_loss, 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# save final dev predictions / ground truths
experiment_data["epochs"]["SPR"]["predictions"] = preds
experiment_data["epochs"]["SPR"]["ground_truth"] = gts

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All metrics saved to", os.path.join(working_dir, "experiment_data.npy"))
