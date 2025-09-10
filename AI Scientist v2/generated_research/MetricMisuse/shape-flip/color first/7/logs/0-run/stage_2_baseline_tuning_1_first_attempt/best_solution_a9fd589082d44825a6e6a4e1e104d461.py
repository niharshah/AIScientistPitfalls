import os, random, string, time, math, copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# --------------------------- I/O & utils ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def complexity_weight(sequence: str) -> int:
    return count_color_variety(sequence) + count_shape_variety(sequence)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [complexity_weight(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return float(sum(correct)) / max(1, sum(weights))


# --------------------------- Data ---------------------------
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
    # ------- synthetic fallback ---------
    print("No SPR_BENCH found; generating synthetic toy dataset")
    shapes, colors = list(string.ascii_uppercase[:6]), list(range(6))

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + str(random.choice(colors)) for _ in range(L)
        )

    def label_rule(seq):  # simple parity rule
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

# Vocab
vocab = {}


def add_token(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
vocab_size, pad_index = len(vocab), len(vocab)  # pad not used


# graph conversion
def seq_to_graph(seq, label):
    tokens = seq.split()
    node_idx = [vocab[tok] for tok in tokens]
    edges = []
    for i in range(len(tokens) - 1):
        edges += [[i, i + 1], [i + 1, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_idx, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def encode_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, (spr["train"], spr["dev"], spr["test"])
)

# loaders
batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# --------------------------- Model ---------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden_dim=64, num_classes=2):
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


model = GNNClassifier(len(vocab), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)


# --------------------------- train / eval helpers ---------------------------
@torch.no_grad()
def eval_loader(loader):
    model.eval()
    tot_loss = tot_correct = tot_samp = 0
    seqs_all, preds_all, ys_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_correct += sum(p == y for p, y in zip(preds, ys))
        tot_samp += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(preds)
        ys_all.extend(ys)
    avg_loss = tot_loss / tot_samp
    acc = tot_correct / tot_samp
    cowa = complexity_weighted_accuracy(seqs_all, ys_all, preds_all)
    return avg_loss, acc, cowa, preds_all, ys_all, seqs_all


def train_one_epoch(loader):
    model.train()
    tot_loss = tot_correct = tot_samp = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_correct += sum(p == y for p, y in zip(preds, ys))
        tot_samp += batch.num_graphs
    return tot_loss / tot_samp, tot_correct / tot_samp


# --------------------------- bookkeeping ---------------------------
experiment_data = {
    "num_epochs": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "sequences": [],
            "best_epoch": None,
        }
    }
}

# --------------------------- training loop with early stopping ---------------------------
max_epochs, patience = 50, 8
best_val_loss = math.inf
pat_cnt = 0
best_state = None
start_time = time.time()

for epoch in range(1, max_epochs + 1):
    tr_loss, tr_acc = train_one_epoch(train_loader)
    val_loss, val_acc, val_cowa, *_ = eval_loader(dev_loader)

    experiment_data["num_epochs"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["num_epochs"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["num_epochs"]["SPR_BENCH"]["metrics"]["train"].append(
        {"acc": tr_acc}
    )
    experiment_data["num_epochs"]["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "cowa": val_cowa}
    )

    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.3f} CoWA={val_cowa:.3f}"
    )

    # early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        experiment_data["num_epochs"]["SPR_BENCH"]["best_epoch"] = epoch
        pat_cnt = 0
    else:
        pat_cnt += 1
        if pat_cnt >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

print(
    f"Training finished in {(time.time()-start_time):.1f}s, best_epoch={experiment_data['num_epochs']['SPR_BENCH']['best_epoch']}"
)

# --------------------------- test evaluation with best model ---------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc, test_cowa, preds, gts, seqs = eval_loader(test_loader)
print(f"TEST -- loss: {test_loss:.4f}, acc: {test_acc:.3f}, CoWA: {test_cowa:.3f}")

experiment_data["num_epochs"]["SPR_BENCH"]["predictions"] = preds
experiment_data["num_epochs"]["SPR_BENCH"]["ground_truth"] = gts
experiment_data["num_epochs"]["SPR_BENCH"]["sequences"] = seqs

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
