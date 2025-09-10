import os, random, string, time, math, copy, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---------- boiler-plate & dirs ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers ----------
def count_color_variety(seq):  # color = token[1:]
    return len({tok[1:] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):  # shape = token[0]
    return len({tok[0] for tok in seq.split() if tok})


def complexity_weight(seq):
    return count_color_variety(seq) + count_shape_variety(seq)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return float(sum(correct)) / max(1, sum(w))


# ---------- load / build dataset ----------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path: str) -> DatasetDict:
    if os.path.isdir(path):

        def _ld(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        print("Loaded real SPR_BENCH from", path)
        return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))
    # synthetic fallback
    print("No SPR_BENCH found, generating toy data")
    shapes, colors = list(string.ascii_uppercase[:6]), list(range(6))

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + str(random.choice(colors)) for _ in range(L)
        )

    def label_rule(seq):
        return sum(int(tok[1:]) for tok in seq.split()) % 2

    def split(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [label_rule(s) for s in seqs],
            }
        )

    return DatasetDict(train=split(1000), dev=split(200), test=split(200))


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# ---------- vocab for token / shape / color ----------
tok_vocab, shape_vocab, color_vocab = {}, {}, {}


def add_tok(tok):
    if tok not in tok_vocab:
        tok_vocab[tok] = len(tok_vocab)
    sh, co = tok[0], tok[1:]
    if sh not in shape_vocab:
        shape_vocab[sh] = len(shape_vocab)
    if co not in color_vocab:
        color_vocab[co] = len(color_vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add_tok(tok)


# ---------- graph builder ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    node_ids = [tok_vocab[t] for t in toks]
    # edges
    e_src, e_dst, e_type = [], [], []
    # type 0: adjacency
    for i in range(n - 1):
        e_src += [i, i + 1]
        e_dst += [i + 1, i]
        e_type += [0, 0]
    # precompute shape/color buckets
    by_shape, by_color = {}, {}
    for idx, t in enumerate(toks):
        by_shape.setdefault(t[0], []).append(idx)
        by_color.setdefault(t[1:], []).append(idx)
    # type1: same shape
    for idxs in by_shape.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    e_src.append(i)
                    e_dst.append(j)
                    e_type.append(1)
    # type2: same color
    for idxs in by_color.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    e_src.append(i)
                    e_dst.append(j)
                    e_type.append(2)
    edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
    edge_type = torch.tensor(e_type, dtype=torch.long)
    x = torch.tensor(node_ids, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


def encode_split(dset):
    return [seq_to_graph(s, l) for s, l in zip(dset["sequence"], dset["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, (spr["train"], spr["dev"], spr["test"])
)

batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ---------- model ----------
class RGCNClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=96, num_classes=2, num_rel=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim)
        self.conv1 = RGCNConv(emb_dim, hidden, num_rel)
        self.conv2 = RGCNConv(hidden, hidden, num_rel)
        self.lin = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, data):
        x, ei, et, batch = data.x, data.edge_index, data.edge_type, data.batch
        x = self.emb(x)
        x = F.relu(self.conv1(x, ei, et))
        x = self.drop(x)
        x = F.relu(self.conv2(x, ei, et))
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = RGCNClassifier(len(tok_vocab), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)


# ---------- train / eval ----------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tloss, correct, total = 0, 0, 0
    seqs_all, preds_all, ys_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        tloss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        correct += sum(p == y for p, y in zip(pred, ys))
        total += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(pred)
        ys_all.extend(ys)
    return (
        tloss / total,
        correct / total,
        complexity_weighted_accuracy(seqs_all, ys_all, preds_all),
        preds_all,
        ys_all,
        seqs_all,
    )


def train_epoch(loader):
    model.train()
    tloss, correct, total = 0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        correct += sum(p == y for p, y in zip(pred, ys))
        total += batch.num_graphs
    return tloss / total, correct / total


# ---------- bookkeeping ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
        "best_epoch": None,
    }
}

# ---------- training loop ----------
max_epochs, patience = 50, 8
best_val_loss, wait, best_state = math.inf, 0, None
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_acc = train_epoch(train_loader)
    val_loss, val_acc, val_cwa, *_ = evaluate(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"acc": tr_acc})
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "CompWA": val_cwa}
    )
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.3f} CompWA={val_cwa:.3f}"
    )
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        experiment_data["SPR_BENCH"]["best_epoch"] = epoch
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# ---------- test ----------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc, test_cwa, preds, gts, seqs = evaluate(test_loader)
print(f"TEST -- loss: {test_loss:.4f}, acc: {test_acc:.3f}, CompWA: {test_cwa:.3f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["sequences"] = seqs

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved logs to", os.path.join(working_dir, "experiment_data.npy"))
