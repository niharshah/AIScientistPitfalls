import os, math, copy, random, time
import numpy as np, torch
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool, BatchNorm

# ------------------------------------------------------------
# housekeeping
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(2024)
random.seed(2024)
np.random.seed(2024)


# ------------------------------------------------------------
# helpers for colours / shapes
# ------------------------------------------------------------
def colour_of(tok):  # everything except first char
    return tok[1:] if len(tok) > 1 else ""


def shape_of(tok):  # first char
    return tok[0] if tok else ""


def count_colour_variety(seq: str) -> int:
    return len({colour_of(t) for t in seq.split()})


def count_shape_variety(seq: str) -> int:
    return len({shape_of(t) for t in seq.split()})


def cswa(seqs, y_true, y_pred):
    w = [count_colour_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


# ------------------------------------------------------------
# dataset loader (same path env var as before)
# ------------------------------------------------------------
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

        print("Loaded SPR_BENCH from", path)
        return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))
    # fallback toy data
    print("SPR_BENCH not found – generating toy data")
    shapes = list("ABCDEF")
    colours = [str(i) for i in range(10)]

    def mk_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colours)
            for _ in range(random.randint(4, 15))
        )

    def lab(seq):  # parity of colour IDs
        return sum(int(colour_of(t) or 0) for t in seq.split()) % 2

    def split(n):
        seqs = [mk_seq() for _ in range(n)]
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": [lab(s) for s in seqs]}
        )

    return DatasetDict(train=split(800), dev=split(200), test=split(200))


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# ------------------------------------------------------------
# build vocabularies over *all* splits -> prevents OOV
# ------------------------------------------------------------
shape_vocab, colour_vocab = {}, {}


def add_shape(s):
    shape_vocab.setdefault(s, len(shape_vocab))


def add_colour(c):
    colour_vocab.setdefault(c, len(colour_vocab))


max_seq_len = 0
for split in ("train", "dev", "test"):
    for seq in spr[split]["sequence"]:
        toks = seq.split()
        max_seq_len = max(max_seq_len, len(toks))
        for t in toks:
            add_shape(shape_of(t))
            add_colour(colour_of(t))

pos_limit = max_seq_len + 1  # safe upper bound
print(f"max sequence length = {max_seq_len},  pos_limit = {pos_limit}")


# ------------------------------------------------------------
# sequence → graph  (no self-loops)
# ------------------------------------------------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shape_ids = torch.tensor([shape_vocab[shape_of(t)] for t in toks], dtype=torch.long)
    colour_ids = torch.tensor(
        [colour_vocab[colour_of(t)] for t in toks], dtype=torch.long
    )
    pos_ids = torch.tensor(list(range(n)), dtype=torch.long)

    # sequential bi-directional edges
    edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    # same-attribute edges
    for i in range(n):
        for j in range(i + 1, n):
            if shape_of(toks[i]) == shape_of(toks[j]):
                edges += [[i, j], [j, i]]
            if colour_of(toks[i]) == colour_of(toks[j]):
                edges += [[i, j], [j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
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

# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------
batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ------------------------------------------------------------
# GNN model
# ------------------------------------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, n_shape, n_col, pos_max, hid=64, n_classes=2, dropout=0.3):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, hid)
        self.col_emb = nn.Embedding(n_col, hid)
        self.pos_emb = nn.Embedding(pos_max, hid)
        self.conv1 = SAGEConv(hid, hid)
        self.bn1 = BatchNorm(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2 = BatchNorm(hid)
        self.lin = nn.Linear(hid, n_classes)
        self.drop = nn.Dropout(dropout)

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
    len(shape_vocab), len(colour_vocab), pos_limit, hid=64, n_classes=num_classes
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ------------------------------------------------------------
# bookkeeping
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# evaluation / training utilities
# ------------------------------------------------------------
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
    acc = tot_correct / tot
    cswa_score = cswa(seqs_all, true_all, pred_all)
    return tot_loss / tot, acc, cswa_score, seqs_all, pred_all, true_all


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


# ------------------------------------------------------------
# training loop
# ------------------------------------------------------------
max_epochs, patience = 40, 8
best_val_loss = math.inf
best_state = None
pat = 0

for epoch in range(1, max_epochs + 1):
    t_start = time.time()
    tr_loss, tr_acc = train_epoch(train_loader)
    val_loss, val_acc, val_cswa, _, _, _ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"epoch": epoch, "acc": tr_acc}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": val_acc, "CSWA": val_cswa}
    )

    print(
        f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  CSWA={val_cswa:.3f}  "
        f"({time.time()-t_start:.1f}s)"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        experiment_data["SPR_BENCH"]["best_epoch"] = epoch
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping.")
            break

# ------------------------------------------------------------
# test evaluation
# ------------------------------------------------------------
if best_state:
    model.load_state_dict(best_state)

test_loss, test_acc, test_cswa, seqs, preds, gts = evaluate(test_loader)
print(f"TEST -- loss:{test_loss:.4f}  acc:{test_acc:.3f}  CSWA:{test_cswa:.3f}")

ed = experiment_data["SPR_BENCH"]
ed["predictions"] = preds
ed["ground_truth"] = gts
ed["sequences"] = seqs

# ------------------------------------------------------------
# save bookkeeping
# ------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
