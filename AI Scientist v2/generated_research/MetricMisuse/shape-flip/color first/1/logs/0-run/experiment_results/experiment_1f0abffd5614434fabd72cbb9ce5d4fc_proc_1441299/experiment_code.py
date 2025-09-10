import os, pathlib, random, string, time, warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ------------------- reproducibility / device -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------- helper metrics ---------------------------
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


# ------------------- dataset utilities -----------------------
def try_load_benchmark():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

    def _load(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_load("train"), dev=_load("dev"), test=_load("test"))


def generate_synthetic(nt=200, nd=60, nte=100):
    shapes, colors = string.ascii_uppercase[:6], string.ascii_lowercase[:6]

    def _gen(n):
        seq, lab = [], []
        for _ in range(n):
            length = random.randint(4, 15)
            toks = [
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            ]
            seq.append(" ".join(toks))
            lab.append(int(toks[0][0] == toks[-1][0]))
        return {"sequence": seq, "label": lab}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        train=Dataset.from_dict(_gen(nt)),
        dev=Dataset.from_dict(_gen(nd)),
        test=Dataset.from_dict(_gen(nte)),
    )


dataset = try_load_benchmark()
if dataset is None:
    print("Benchmark not found – generating synthetic data.")
    dataset = generate_synthetic()

num_classes = len(set(dataset["train"]["label"]))
print(
    f"Train {len(dataset['train'])}, Dev {len(dataset['dev'])}, "
    f"Test {len(dataset['test'])}, num_classes={num_classes}"
)

# ------------------- vocabulary ------------------------------
vocab = {}
for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)


# ------------------- graph construction ----------------------
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

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=64)
test_loader = DataLoader(test_graphs, batch_size=64)


# ------------------- model -----------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden=64, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ------------------- training utility ------------------------
def run_training(weight_decay, epochs=5):
    model = GNNClassifier(len(vocab), num_classes=num_classes).to(device)
    optim = Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    cri = nn.CrossEntropyLoss()

    record = {
        "weight_decay": weight_decay,
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, epochs + 1):
        # --- train
        model.train()
        tloss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = cri(out, batch.y)
            loss.backward()
            optim.step()
            tloss += loss.item() * batch.num_graphs
        tloss /= len(train_loader.dataset)
        record["losses"]["train"].append(tloss)

        # --- validation
        model.eval()
        vloss = 0.0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = cri(logits, batch.y)
                vloss += loss.item() * batch.num_graphs
                preds.extend(logits.argmax(-1).cpu().tolist())
                gts.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq_raw)
        vloss /= len(dev_loader.dataset)
        record["losses"]["val"].append(vloss)

        acc = np.mean([p == t for p, t in zip(preds, gts)])
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        caa = complexity_adjusted_accuracy(seqs, gts, preds)
        record["metrics"]["val"].append(
            {"acc": acc, "cwa": cwa, "swa": swa, "caa": caa}
        )

        print(
            f"wd={weight_decay:>5}: epoch {ep} | "
            f"train_loss {tloss:.4f} | val_loss {vloss:.4f} | acc {acc:.3f}"
        )

    record["predictions"] = preds
    record["ground_truth"] = gts
    return record


# ------------------- hyperparameter sweep --------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
experiment_data = {"weight_decay": {"SPR": {}}}

for wd in weight_decays:
    result = run_training(wd)
    experiment_data["weight_decay"]["SPR"][f"wd_{wd}"] = result

# ------------------- save ------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "All runs finished. Data saved to", os.path.join(working_dir, "experiment_data.npy")
)
