# No-Batch-Normalization ablation study
import os, time, math, copy, random
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool
from torch_geometric.utils import add_self_loops

# ---------------- bookkeeping dict ----------------
experiment_data = {
    "no_batch_norm": {
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

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================================================================
# 1. helpers ------------------------------------------------------------
def colour_of(token: str) -> str:
    return token[1:] if len(token) > 1 else ""


def shape_of(token: str) -> str:
    return token[0]


def count_colour_variety(seq: str) -> int:
    return len(set(colour_of(tok) for tok in seq.split() if tok))


def count_shape_variety(seq: str) -> int:
    return len(set(shape_of(tok) for tok in seq.split() if tok))


def complexity_weight(seq: str) -> int:
    return count_colour_variety(seq) + count_shape_variety(seq)


def comp_weighted_acc(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1, sum(w))


# ======================================================================
# 2. dataset ------------------------------------------------------------
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

        print("Loaded SPR_BENCH from", path)
        return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))

    print("SPR_BENCH not found – generating toy data")
    shapes = list("ABCDEF")
    colours = [str(i) for i in range(10)]

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colours) for _ in range(L)
        )

    def label_rule(seq):
        return sum(int(colour_of(tok)) for tok in seq.split()) % 2

    def split(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [label_rule(s) for s in seqs],
            }
        )

    return DatasetDict(train=split(800), dev=split(200), test=split(200))


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# ======================================================================
# 3. vocab --------------------------------------------------------------
shape_vocab, colour_vocab = {}, {}


def add_shape(s):
    if s not in shape_vocab:
        shape_vocab[s] = len(shape_vocab)


def add_colour(c):
    if c not in colour_vocab:
        colour_vocab[c] = len(colour_vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add_shape(shape_of(tok))
        add_colour(colour_of(tok))
pos_limit = 20


# ======================================================================
# 4. seq -> graph -------------------------------------------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shape_ids = torch.tensor([shape_vocab[shape_of(t)] for t in toks], dtype=torch.long)
    colour_ids = torch.tensor(
        [colour_vocab[colour_of(t)] for t in toks], dtype=torch.long
    )
    pos_ids = torch.tensor(list(range(n)), dtype=torch.long)

    edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    for i in range(n):
        for j in range(i + 1, n):
            if shape_of(toks[i]) == shape_of(toks[j]):
                edges += [[i, j], [j, i]]
            if colour_of(toks[i]) == colour_of(toks[j]):
                edges += [[i, j], [j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)

    return Data(
        shape_id=shape_ids,
        colour_id=colour_ids,
        pos_id=pos_ids,
        edge_index=edge_index,
        y=torch.tensor(label, dtype=torch.long),
        seq=seq,
    )


def encode_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, (spr["train"], spr["dev"], spr["test"])
)

# ======================================================================
# 5. loaders ------------------------------------------------------------
batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ======================================================================
# 6. model --------------------------------------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, n_shape, n_colour, pos_max, hid=64, n_classes=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, hid)
        self.col_emb = nn.Embedding(n_colour, hid)
        self.pos_emb = nn.Embedding(pos_max, hid)

        self.conv1 = SAGEConv(hid, hid)
        self.bn1 = nn.Identity()  # BatchNorm removed
        self.conv2 = SAGEConv(hid, hid)
        self.bn2 = nn.Identity()  # BatchNorm removed

        self.lin = nn.Linear(hid, n_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, data):
        x = (
            self.shape_emb(data.shape_id)
            + self.col_emb(data.colour_id)
            + self.pos_emb(data.pos_id)
        )
        x = torch.relu(self.bn1(self.conv1(x, data.edge_index)))
        x = self.drop(x)
        x = torch.relu(self.bn2(self.conv2(x, data.edge_index)))
        graph_x = global_add_pool(x, data.batch)
        return self.lin(graph_x)


model = GNNClassifier(
    len(shape_vocab), len(colour_vocab), pos_limit, 64, num_classes
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# shorthand for dict access
exp = experiment_data["no_batch_norm"]["SPR_BENCH"]


# ======================================================================
# 7. eval & train helpers ----------------------------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss = tot_correct = tot = 0
    seqs_all, pred_all, true_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        gts = batch.y.cpu().tolist()
        tot_correct += sum(p == g for p, g in zip(preds, gts))
        tot += batch.num_graphs

        seqs_all.extend(batch.seq)
        pred_all.extend(preds)
        true_all.extend(gts)
    return (
        tot_loss / tot,
        tot_correct / tot,
        comp_weighted_acc(seqs_all, true_all, pred_all),
        seqs_all,
        pred_all,
        true_all,
    )


def train_epoch(loader):
    model.train()
    tot_loss = tot_correct = tot = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        gts = batch.y.cpu().tolist()
        tot_correct += sum(p == g for p, g in zip(preds, gts))
        tot += batch.num_graphs
    return tot_loss / tot, tot_correct / tot


# ======================================================================
# 8. training loop ------------------------------------------------------
max_epochs, patience = 40, 8
best_val_loss = math.inf
best_state = None
pat = 0

for epoch in range(1, max_epochs + 1):
    tr_loss, tr_acc = train_epoch(train_loader)
    val_loss, val_acc, val_cwa, _, _, _ = evaluate(dev_loader)

    exp["losses"]["train"].append(tr_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["train"].append({"epoch": epoch, "acc": tr_acc})
    exp["metrics"]["val"].append({"epoch": epoch, "acc": val_acc, "CompWA": val_cwa})

    print(
        f"Epoch {epoch:02d}: val_loss={val_loss:.4f} val_acc={val_acc:.3f} CompWA={val_cwa:.3f}"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        exp["best_epoch"] = epoch
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping.")
            break

# ======================================================================
# 9. test ---------------------------------------------------------------
if best_state:
    model.load_state_dict(best_state)
test_loss, test_acc, test_cwa, seqs, preds, gts = evaluate(test_loader)
print(f"TEST -- loss:{test_loss:.4f} acc:{test_acc:.3f} CompWA:{test_cwa:.3f}")

exp["predictions"] = preds
exp["ground_truth"] = gts
exp["sequences"] = seqs
exp["metrics"]["test"] = {"acc": test_acc, "CompWA": test_cwa}
exp["losses"]["test"] = test_loss

# ======================================================================
# 10. save --------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("saved experiment_data.npy")
