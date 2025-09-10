import os, math, time, copy, random, string, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

# --------------------------------------------------- boiler-plate & paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------- metrics
def count_color_variety(seq: str) -> int:
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weight(seq: str) -> int:
    return count_color_variety(seq) + count_shape_variety(seq)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return float(sum(cor)) / max(1, sum(w))


# --------------------------------------------------- dataset helper
def load_spr(path: str = "./SPR_BENCH") -> DatasetDict:
    if os.path.isdir(path):

        def _csv(name):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{name}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        print("Loaded real SPR_BENCH")
        return DatasetDict(train=_csv("train"), dev=_csv("dev"), test=_csv("test"))
    # tiny synthetic fallback
    shapes, colours = list(string.ascii_uppercase[:6]), list(range(6))

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + str(random.choice(colours)) for _ in range(L)
        )

    def rule(s):
        return sum(int(tok[1:]) for tok in s.split()) % 2

    def split(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": [rule(x) for x in seqs]}
        )

    print("Generated synthetic SPR")
    return DatasetDict(train=split(800), dev=split(200), test=split(200))


spr = load_spr(os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH"))
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# --------------------------------------------------- mappings
shape2idx = {
    s: i
    for i, s in enumerate(
        sorted({t[0] for t in spr["train"]["sequence"] for t in t.split()})
    )
}
colour2idx = {
    c: i
    for i, c in enumerate(
        sorted({t[1:] for t in spr["train"]["sequence"] for t in t.split()})
    )
}
max_pos = max(len(seq.split()) for seq in spr["train"]["sequence"]) + 1


def seq_to_graph(seq: str, label: int):
    toks = seq.split()
    n = len(toks)
    shape_ids = torch.tensor([shape2idx[t[0]] for t in toks], dtype=torch.long)
    colour_ids = torch.tensor([colour2idx[t[1:]] for t in toks], dtype=torch.long)
    pos_ids = torch.arange(n, dtype=torch.long)
    # bidirectional sequential edges
    edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T.contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        shape_id=shape_ids,
        colour_id=colour_ids,
        pos_id=pos_ids,
        edge_index=edge_index,
        y=torch.tensor([label]),
        seq=seq,
    )


def encode_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, (spr["train"], spr["dev"], spr["test"])
)
train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=256, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=256, shuffle=False)


# --------------------------------------------------- model
class FactorGNN(nn.Module):
    def __init__(self, n_shape, n_colour, max_pos, hid=96, out_dim=2, emb=32):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb)
        self.col_emb = nn.Embedding(n_colour, emb)
        self.pos_emb = nn.Embedding(max_pos + 1, emb)
        self.conv1 = SAGEConv(emb * 3, hid)
        self.bn1 = BatchNorm(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2 = BatchNorm(hid)
        self.lin = nn.Linear(hid, out_dim)

    def forward(self, data):
        x = torch.cat(
            [
                self.shape_emb(data.shape_id),
                self.col_emb(data.colour_id),
                self.pos_emb(
                    torch.clamp(data.pos_id, max=self.pos_emb.num_embeddings - 1)
                ),
            ],
            dim=-1,
        )
        x = F.relu(self.bn1(self.conv1(x, data.edge_index)))
        x = F.relu(self.bn2(self.conv2(x, data.edge_index)))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = FactorGNN(
    len(shape2idx), len(colour2idx), max_pos, hid=128, out_dim=num_classes
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# --------------------------------------------------- bookkeeping
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


# --------------------------------------------------- helpers
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tloss = tcorr = tsamp = 0
    seqs_all, preds_all, ys_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        tloss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tcorr += sum(p == y for p, y in zip(preds, ys))
        tsamp += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(preds)
        ys_all.extend(ys)
    return (
        tloss / tsamp,
        tcorr / tsamp,
        comp_weighted_accuracy(seqs_all, ys_all, preds_all),
        preds_all,
        ys_all,
        seqs_all,
    )


def train_epoch(loader):
    model.train()
    tloss = tcorr = tsamp = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tcorr += sum(p == y for p, y in zip(preds, ys))
        tsamp += batch.num_graphs
    return tloss / tsamp, tcorr / tsamp


# --------------------------------------------------- training loop
best_val = math.inf
patience = 8
wait = 0
best_state = None
start = time.time()
for epoch in range(1, 51):
    tr_loss, tr_acc = train_epoch(train_loader)
    val_loss, val_acc, val_cwa, *_ = evaluate(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"acc": tr_acc})
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "CompWA": val_cwa}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.3f} | CompWA={val_cwa:.3f}"
    )
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        experiment_data["SPR_BENCH"]["best_epoch"] = epoch
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# --------------------------------------------------- test evaluation
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc, test_cwa, preds, gts, seqs = evaluate(test_loader)
print(f"TEST -- loss:{test_loss:.4f} acc:{test_acc:.3f} CompWA:{test_cwa:.3f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["sequences"] = seqs
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
