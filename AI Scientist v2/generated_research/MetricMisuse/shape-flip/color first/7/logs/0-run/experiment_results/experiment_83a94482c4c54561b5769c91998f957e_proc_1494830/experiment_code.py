import os, math, time, random, string, copy
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helpers ----------
def count_color_variety(seq):
    return len(set(tok[1:] for tok in seq.split()))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def comp_weight(seq):
    return count_color_variety(seq) + count_shape_variety(seq)


def compWA(seqs, y, yp):
    w = [comp_weight(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y, yp)]
    return sum(c) / max(1, sum(w))


# ---------- load / fallback dataset ----------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path):
    if os.path.isdir(path):

        def _ld(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        print("Loading real SPR_BENCH")
        return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))
    # synthetic tiny fallback
    shapes, colours = list(string.ascii_uppercase[:6]), list(map(str, range(6)))

    def mk():
        return " ".join(
            random.choice(shapes) + random.choice(colours)
            for _ in range(random.randint(4, 9))
        )

    def lab(s):
        return sum(int(tok[1:]) for tok in s.split()) % 2

    def build(n):
        seq = [mk() for _ in range(n)]
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seq, "label": [lab(s) for s in seq]}
        )

    print("SPR_BENCH not found – generating toy data")
    return DatasetDict(train=build(600), dev=build(150), test=build(150))


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))

# ---------- vocab ----------
vocab = {}


def add(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add(tok)

# ---------- graph conversion ----------
from torch_geometric.data import Data


def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    node_ids = [vocab[t] for t in toks]
    edges = []
    # sequential edges
    for i in range(n - 1):
        edges += ((i, i + 1), (i + 1, i))
    # colour / shape edges
    col_groups, shp_groups = {}, {}
    for i, t in enumerate(toks):
        col_groups.setdefault(t[1:], []).append(i)
        shp_groups.setdefault(t[0], []).append(i)
    for grp in (col_groups, shp_groups):
        for idxs in grp.values():
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edges.append((i, j))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.tensor(node_ids),
        edge_index=edge_index,
        y=torch.tensor([label]),
        seq=seq,
    )


def enc_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_g, dev_g, test_g = map(enc_split, (spr["train"], spr["dev"], spr["test"]))

from torch_geometric.loader import DataLoader

batch_size = 128
train_loader = DataLoader(train_g, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_g, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_g, batch_size=batch_size, shuffle=False)

# ---------- model ----------
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNN(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=64, cls=num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb)
        self.conv1 = SAGEConv(emb, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, cls)

    def forward(self, data):
        x = self.emb(data.x.to(device))
        x = F.relu(self.conv1(x, data.edge_index.to(device)))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch.to(device))
        return self.lin(x)


model = GNN(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# ---------- exp data store ----------
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


# ---------- train / eval ----------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tl = tc = ts = 0
    seqs = []
    ys = []
    yp = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        tl += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        y = batch.y.cpu().tolist()
        tc += sum(p == t for p, t in zip(pred, y))
        ts += batch.num_graphs
        seqs += batch.seq
        ys += y
        yp += pred
    return (tl / ts, tc / ts, compWA(seqs, ys, yp), yp, ys, seqs)


def train_one_epoch(loader):
    model.train()
    tl = tc = ts = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tl += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        y = batch.y.cpu().tolist()
        tc += sum(p == t for p, t in zip(pred, y))
        ts += batch.num_graphs
    return tl / ts, tc / ts


# ---------- training loop ----------
max_epochs, patience = 30, 5
best_val_loss = math.inf
pat = 0
best_state = None
start = time.time()
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_acc = train_one_epoch(train_loader)
    val_loss, val_acc, val_cwa, *_ = evaluate(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"acc": tr_acc})
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "compWA": val_cwa}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}  acc={val_acc:.3f}  CompWA={val_cwa:.3f}"
    )
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        experiment_data["SPR_BENCH"]["best_epoch"] = epoch
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping")
            break

# ---------- test ----------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc, test_cwa, preds, gts, seqs = evaluate(test_loader)
print(f"TEST -- loss:{test_loss:.4f}  acc:{test_acc:.3f}  CompWA:{test_cwa:.3f}")

exp = experiment_data["SPR_BENCH"]
exp["predictions"], exp["ground_truth"], exp["sequences"] = preds, gts, seqs
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Metrics saved to", os.path.join(working_dir, "experiment_data.npy"))
