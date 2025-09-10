import os, math, random, string, time, copy, numpy as np, torch, torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict

# -------------------- working dir & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- metric helpers --------------------------
def count_color_variety(seq):  # number of unique colours
    return len(set(tok[1:] for tok in seq.split()))


def count_shape_variety(seq):  # number of unique shapes
    return len(set(tok[0] for tok in seq.split()))


def comp_weight(seq):  # complexity = colours + shapes
    return count_color_variety(seq) + count_shape_variety(seq)


def compWA(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1, sum(w))


# -------------------- load or fabricate SPR_BENCH --------------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path):
    if os.path.isdir(path):
        print("Loading real SPR_BENCH from", path)

        def _ld(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))
    # fallback tiny synthetic dataset
    shapes, colours = list(string.ascii_uppercase[:6]), list(map(str, range(6)))

    def make_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colours)
            for _ in range(random.randint(4, 10))
        )

    def label(seq):  # parity on colour ids
        return sum(int(tok[1:]) for tok in seq.split()) % 2

    def build(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": [label(s) for s in seqs]}
        )

    print("SPR_BENCH not found – generating synthetic fallback")
    return DatasetDict(train=build(800), dev=build(200), test=build(200))


spr = load_spr(SPR_PATH)

num_classes = len(set(spr["train"]["label"]))

# -------------------- build vocabularies ----------------------
shape_vocab, colour_vocab = {}, {}


def add(d, k):
    if k not in d:
        d[k] = len(d)


max_pos = 0
for seq in spr["train"]["sequence"]:
    toks = seq.split()
    max_pos = max(max_pos, len(toks))
    for tok in toks:
        add(shape_vocab, tok[0])
        add(colour_vocab, tok[1:])

max_pos = min(max_pos, 30)  # cap position embedding size

# -------------------- graphs via torch_geometric --------------
from torch_geometric.data import Data


def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    s_ids = [shape_vocab[t[0]] for t in toks]
    c_ids = [colour_vocab[t[1:]] for t in toks]
    p_ids = (
        list(range(min(n, max_pos))) + [max_pos - 1] * (n - max_pos)
        if n > max_pos
        else list(range(n))
    )

    edges, etype = [], []  # edge types: 0-sequential, 1-colour, 2-shape
    # sequential
    for i in range(n - 1):
        edges += [(i, i + 1), (i + 1, i)]
        etype += [0, 0]
    # colour
    col_groups = {}
    for i, cid in enumerate(c_ids):
        col_groups.setdefault(cid, []).append(i)
    for idxs in col_groups.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append((i, j))
                    etype.append(1)
    # shape
    shp_groups = {}
    for i, sid in enumerate(s_ids):
        shp_groups.setdefault(sid, []).append(i)
    for idxs in shp_groups.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append((i, j))
                    etype.append(2)

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = (
        torch.tensor(etype, dtype=torch.long)
        if etype
        else torch.empty((0,), dtype=torch.long)
    )
    return Data(
        sid=torch.tensor(s_ids, dtype=torch.long),
        cid=torch.tensor(c_ids, dtype=torch.long),
        pid=torch.tensor(p_ids, dtype=torch.long),
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label], dtype=torch.long),
        seq=seq,
    )


def encode_split(ds):  # to list[Data]
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_g, dev_g, test_g = map(encode_split, (spr["train"], spr["dev"], spr["test"]))

from torch_geometric.loader import DataLoader

batch_size = 128
train_loader = DataLoader(train_g, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_g, batch_size=batch_size)
test_loader = DataLoader(test_g, batch_size=batch_size)

# -------------------- relational GNN model --------------------
from torch_geometric.nn import RGCNConv, global_mean_pool


class RelGNN(nn.Module):
    def __init__(self, n_shape, n_colour, n_pos, hid=64, out=num_classes):
        super().__init__()
        emb = 32
        self.shape_emb = nn.Embedding(n_shape, emb)
        self.col_emb = nn.Embedding(n_colour, emb)
        self.pos_emb = nn.Embedding(n_pos, emb)
        self.rgcn1 = RGCNConv(emb, hid, num_relations=3)
        self.rgcn2 = RGCNConv(hid, hid, num_relations=3)
        self.lin = nn.Linear(hid, out)

    def forward(self, data):
        x = self.shape_emb(data.sid) + self.col_emb(data.cid) + self.pos_emb(data.pid)
        x = F.relu(self.rgcn1(x, data.edge_index, data.edge_type))
        x = F.relu(self.rgcn2(x, data.edge_index, data.edge_type))
        x = global_mean_pool(x, data.batch)  # average over nodes per graph
        return self.lin(x)


model = RelGNN(len(shape_vocab), len(colour_vocab), max_pos).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-3)

# -------------------- experiment data store -------------------
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


# -------------------- training / evaluation -------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss = tot_correct = tot_items = 0
    all_seq, all_y, all_pred = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_correct += sum(p == t for p, t in zip(preds, ys))
        tot_items += batch.num_graphs
        all_seq += batch.seq
        all_y += ys
        all_pred += preds
    return (
        tot_loss / tot_items,
        tot_correct / tot_items,
        compWA(all_seq, all_y, all_pred),
        all_pred,
        all_y,
        all_seq,
    )


def train_one_epoch(loader):
    model.train()
    tot_loss = tot_correct = tot_items = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_correct += sum(p == t for p, t in zip(preds, ys))
        tot_items += batch.num_graphs
    return tot_loss / tot_items, tot_correct / tot_items


# -------------------- main training loop ----------------------
max_epochs, patience = 40, 6
best_val_loss = math.inf
pat = 0
best_state = None
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_acc = train_one_epoch(train_loader)
    val_loss, val_acc, val_comp, *_ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"acc": tr_acc})
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "CompWA": val_comp}
    )

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}  acc={val_acc:.3f}  CompWA={val_comp:.3f}"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        experiment_data["SPR_BENCH"]["best_epoch"] = epoch
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping triggered")
            break

# -------------------- final test ------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc, test_comp, preds, gts, seqs = evaluate(test_loader)
print(f"TEST -- loss:{test_loss:.4f}  acc:{test_acc:.3f}  CompWA:{test_comp:.3f}")

exp = experiment_data["SPR_BENCH"]
exp["predictions"], exp["ground_truth"], exp["sequences"] = preds, gts, seqs

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
