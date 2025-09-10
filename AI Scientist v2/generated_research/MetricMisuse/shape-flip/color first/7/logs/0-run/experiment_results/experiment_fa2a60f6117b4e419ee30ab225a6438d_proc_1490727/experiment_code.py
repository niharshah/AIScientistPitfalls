import os, random, string, time, json, math, gc
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------- experiment bookkeeping ----------
experiment_data = {
    "emb_dim_tuning": {
        "SPR_BENCH": {"runs": []}  # list of dicts, one per emb_dim value
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- helper functions ----------
def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def complexity_weight(s: str) -> int:
    return count_color_variety(s) + count_shape_variety(s)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return float(sum(corr)) / max(1, sum(w))


# ---------- load or create dataset ----------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path: str) -> DatasetDict:
    if os.path.isdir(path):

        def _csv(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        print("Loading real SPR_BENCH from", path)
        return DatasetDict(train=_csv("train"), dev=_csv("dev"), test=_csv("test"))
    # synthetic fallback
    print("No SPR_BENCH found; generating synthetic toy dataset")
    shapes = list(string.ascii_uppercase[:6])
    colors = list(range(6))

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + str(random.choice(colors)) for _ in range(L)
        )

    def label_rule(seq):
        return sum(int(tok[1]) for tok in seq.split()) % 2

    def build_split(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [label_rule(s) for s in seqs],
            }
        )

    return DatasetDict(
        train=build_split(800), dev=build_split(200), test=build_split(200)
    )


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# ---------- vocabulary ----------
vocab = {}


def add_tok(t):
    if t not in vocab:
        vocab[t] = len(vocab)


for seq in spr["train"]["sequence"]:
    for t in seq.split():
        add_tok(t)
vocab_size = len(vocab)


def seq_to_graph(seq, label):
    toks = seq.split()
    node_idx = [vocab[t] for t in toks]
    edges = []
    for i in range(len(toks) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(
        x=torch.tensor(node_idx, dtype=torch.long),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        seq=seq,
    )


def encode(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode, (spr["train"], spr["dev"], spr["test"])
)

# ---------- data loaders ----------
BATCH_SIZE = 128
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)


# ---------- model ----------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.emb(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------- train / eval ----------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = tot_corr = tot_samples = 0
    seqs_all, preds_all, labels_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_corr += sum(int(a == b) for a, b in zip(pred, ys))
        tot_samples += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(pred)
        labels_all.extend(ys)
    avg_loss = total_loss / max(1, tot_samples)
    acc = tot_corr / max(1, tot_samples)
    cowa = complexity_weighted_accuracy(seqs_all, labels_all, preds_all)
    return avg_loss, acc, cowa, preds_all, labels_all, seqs_all


# ---------- hyperparameter sweep ----------
EPOCHS = 5
HIDDEN_DIM = 64
emb_dims = [32, 64, 128, 256]

for emb_dim in emb_dims:
    print(f"\n=== Training with emb_dim={emb_dim} ===")
    model = GNNClassifier(len(vocab), emb_dim, HIDDEN_DIM, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    run_record = {
        "emb_dim": emb_dim,
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_cowa, *_ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, val_cowa, *_ = run_epoch(model, dev_loader, criterion, None)
        run_record["losses"]["train"].append(tr_loss)
        run_record["losses"]["val"].append(val_loss)
        run_record["metrics"]["train"].append({"acc": tr_acc, "cowa": tr_cowa})
        run_record["metrics"]["val"].append({"acc": val_acc, "cowa": val_cowa})
        print(
            f"epoch {epoch}: val_loss={val_loss:.4f} acc={val_acc:.3f} cowa={val_cowa:.3f}"
        )
    # final test
    test_loss, test_acc, test_cowa, preds, gts, seqs = run_epoch(
        model, test_loader, criterion, None
    )
    print(
        f"TEST emb_dim={emb_dim}: loss={test_loss:.4f} acc={test_acc:.3f} cowa={test_cowa:.3f}"
    )
    run_record.update(
        {
            "test": {"loss": test_loss, "acc": test_acc, "cowa": test_cowa},
            "predictions": preds,
            "ground_truth": gts,
            "sequences": seqs,
        }
    )
    experiment_data["emb_dim_tuning"]["SPR_BENCH"]["runs"].append(run_record)
    # clean up
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
