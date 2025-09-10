import os, math, time, random, string, copy
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict

# ---------------- working dir & device ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- helper metrics ----------------------
def count_color_variety(seq):
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def comp_weight(seq):
    return count_color_variety(seq) + count_shape_variety(seq)


def compWA(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    correct = [wt if yp == yt else 0 for wt, yp, yt in zip(w, y_pred, y_true)]
    return sum(correct) / max(1, sum(w))


# ---------------- data loading / fallback -------------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def build_toy():
    shapes = list(string.ascii_uppercase[:6])
    colours = [str(i) for i in range(6)]

    def mkseq():
        return " ".join(
            random.choice(shapes) + random.choice(colours)
            for _ in range(random.randint(4, 9))
        )

    def lab(seq):  # dummy rule
        return sum(int(tok[1:]) for tok in seq.split()) % 2

    def ds(n):
        seqs = [mkseq() for _ in range(n)]
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": [lab(s) for s in seqs]}
        )

    print("SPR_BENCH not found: generating synthetic toy data")
    return DatasetDict(train=ds(800), dev=ds(200), test=ds(200))


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
    return build_toy()


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))

# ---------------- vocabularies ------------------------
shape_vocab, colour_vocab = {}, {}


def add_vocab(tok):
    s, c = tok[0], tok[1:]
    if s not in shape_vocab:
        shape_vocab[s] = len(shape_vocab)
    if c not in colour_vocab:
        colour_vocab[c] = len(colour_vocab)


for seq in spr["train"]["sequence"]:
    for t in seq.split():
        add_vocab(t)

# ---------------- graph building ----------------------
from torch_geometric.data import Data


def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shape_ids = [shape_vocab[t[0]] for t in toks]
    colour_ids = [colour_vocab[t[1:]] for t in toks]
    pos_ids = list(range(n))

    edges, etypes = [], []
    # order edges
    for i in range(n - 1):
        edges += [(i, i + 1), (i + 1, i)]
        etypes += [0, 0]
    # same-colour edges
    col_groups = {}
    for i, t in enumerate(toks):
        col_groups.setdefault(t[1:], []).append(i)
    for idxs in col_groups.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append((i, j))
                    etypes.append(1)
    # same-shape edges
    shp_groups = {}
    for i, t in enumerate(toks):
        shp_groups.setdefault(t[0], []).append(i)
    for idxs in shp_groups.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append((i, j))
                    etypes.append(2)

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = (
        torch.tensor(etypes, dtype=torch.long)
        if etypes
        else torch.empty((0,), dtype=torch.long)
    )

    return Data(
        shape_id=torch.tensor(shape_ids, dtype=torch.long),
        colour_id=torch.tensor(colour_ids, dtype=torch.long),
        pos_id=torch.tensor(pos_ids, dtype=torch.long),
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label], dtype=torch.long),
        seq=seq,
    )


def enc_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_g, dev_g, test_g = map(enc_split, (spr["train"], spr["dev"], spr["test"]))

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_g, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_g, batch_size=128)
test_loader = DataLoader(test_g, batch_size=128)

# ---------------- model -------------------------------
from torch_geometric.nn import RGCNConv, global_mean_pool


class FactorisedRGCN(nn.Module):
    def __init__(
        self, n_shape, n_colour, n_pos=50, dim=64, hid=96, n_rel=3, n_class=num_classes
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, dim)
        self.col_emb = nn.Embedding(n_colour, dim)
        self.pos_emb = nn.Embedding(n_pos, dim)
        self.conv1 = RGCNConv(dim, hid, num_relations=n_rel)
        self.bn1 = nn.BatchNorm1d(hid)
        self.conv2 = RGCNConv(hid, hid, num_relations=n_rel)
        self.bn2 = nn.BatchNorm1d(hid)
        self.lin = nn.Linear(hid, n_class)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        # build node embedding on the fly
        pos_ids = data.pos_id.clamp(max=self.pos_emb.num_embeddings - 1)
        x = (
            self.shape_emb(data.shape_id)
            + self.col_emb(data.colour_id)
            + self.pos_emb(pos_ids)
        )
        x = F.relu(self.bn1(self.conv1(x, data.edge_index, data.edge_type)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, data.edge_index, data.edge_type)))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = FactorisedRGCN(len(shape_vocab), len(colour_vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ---------------- experiment data store ----------------
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


# ---------------- training / evaluation ----------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = correct = tot = 0
    ys, yps, seqs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        gold = batch.y.cpu().tolist()
        correct += sum(p == g for p, g in zip(pred, gold))
        tot += batch.num_graphs
        ys += gold
        yps += pred
        seqs += batch.seq
    return (total_loss / tot, correct / tot, compWA(seqs, ys, yps), yps, ys, seqs)


def train_one_epoch(loader):
    model.train()
    tloss = cor = tot = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        gold = batch.y.cpu().tolist()
        cor += sum(p == g for p, g in zip(pred, gold))
        tot += batch.num_graphs
    return tloss / tot, cor / tot


# ---------------- main training loop -------------------
best_val = math.inf
patience, pat = 6, 0
best_state = None
for epoch in range(1, 40 + 1):
    tr_loss, tr_acc = train_one_epoch(train_loader)
    val_loss, val_acc, val_cw, *_ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"acc": tr_acc})
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "CompWA": val_cw}
    )

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}  acc={val_acc:.3f}  CompWA={val_cw:.3f}"
    )
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        experiment_data["SPR_BENCH"]["best_epoch"] = epoch
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping triggered")
            break

# ---------------- test evaluation ----------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc, test_cw, preds, gts, seqs = evaluate(test_loader)
print(f"TEST -- loss:{test_loss:.4f}  acc:{test_acc:.3f}  CompWA:{test_cw:.3f}")

exp = experiment_data["SPR_BENCH"]
exp["predictions"], exp["ground_truth"], exp["sequences"] = preds, gts, seqs
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
