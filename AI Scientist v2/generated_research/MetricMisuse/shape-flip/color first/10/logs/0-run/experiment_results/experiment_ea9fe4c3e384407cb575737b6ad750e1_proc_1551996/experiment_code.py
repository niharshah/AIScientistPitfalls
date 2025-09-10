import os, pathlib, random, numpy as np, torch, time
from torch import nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from datasets import DatasetDict, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# --------------------------- book-keeping ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------------------- device ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- metrics ---------------------------
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split()))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y_t, y_p)]
    return sum(c) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y_t, y_p)]
    return sum(c) / (sum(w) or 1)


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y_t, y_p)]
    return sum(c) / (sum(w) or 1)


# --------------------------- data ---------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # fallback tiny synthetic

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labs = [], []
        for i in range(n):
            L = random.randint(4, 8)
            tok = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seqs.append(" ".join(tok))
            labs.append(random.choice(["yes", "no"]))
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    spr = DatasetDict(
        {
            "train": Dataset.from_dict(synth(800)),
            "dev": Dataset.from_dict(synth(200)),
            "test": Dataset.from_dict(synth(200)),
        }
    )

shapes = set(
    ch for s in spr["train"]["sequence"] for ch in [tok[0] for tok in s.split()]
)
colors = set(
    cl for s in spr["train"]["sequence"] for cl in [tok[1] for tok in s.split()]
)
shape2idx = {s: i for i, s in enumerate(sorted(shapes))}
color2idx = {c: i for i, c in enumerate(sorted(colors))}
labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}


def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shp = [shape2idx[t[0]] for t in toks]
    col = [color2idx[t[1]] for t in toks]
    if n == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return Data(
        shape=torch.tensor(shp),
        color=torch.tensor(col),
        edge_index=edge_index,
        y=torch.tensor([label2idx[label]]),
    )


def build_graphs(split):
    return [
        seq_to_graph(s, l) for s, l in zip(spr[split]["sequence"], spr[split]["label"])
    ]


graphs = {split: build_graphs(split) for split in ["train", "dev", "test"]}


# --------------------------- model ---------------------------
class GNNClassifier(nn.Module):
    def __init__(self, n_shape, n_color, emb_dim=32, hid=64, num_cls=2):
        super().__init__()
        self.emb_shape = nn.Embedding(n_shape, emb_dim)
        self.emb_color = nn.Embedding(n_color, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, num_cls)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x = self.emb_shape(data.shape) + self.emb_color(data.color)
        x = self.conv1(x, data.edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# --------------------------- loaders ---------------------------
train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
dev_loader = DataLoader(graphs["dev"], batch_size=128)
test_loader = DataLoader(graphs["test"], batch_size=128)


# --------------------------- train & eval ---------------------------
def evaluate(model, loader, seqs):
    model.eval()
    all_p, all_t, all_s = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for i, b in enumerate(loader):
            seq_batch = seqs[
                i * loader.batch_size : i * loader.batch_size + b.num_graphs
            ]
            b = b.to(device)
            out = model(b)
            loss = cross_entropy(out, b.y)
            loss_sum += loss.item() * b.num_graphs
            preds = out.argmax(-1).cpu().tolist()
            gold = b.y.cpu().tolist()
            all_p.extend(preds)
            all_t.extend(gold)
            all_s.extend(seq_batch)
    loss_avg = loss_sum / len(loader.dataset)
    pred_lbl = [labels[p] for p in all_p]
    true_lbl = [labels[t] for t in all_t]
    cwa = color_weighted_accuracy(all_s, true_lbl, pred_lbl)
    swa = shape_weighted_accuracy(all_s, true_lbl, pred_lbl)
    cpx = complexity_weighted_accuracy(all_s, true_lbl, pred_lbl)
    return loss_avg, cwa, swa, cpx, pred_lbl, true_lbl


# --------------------------- training loop ---------------------------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
model = GNNClassifier(len(shape2idx), len(color2idx), num_cls=len(labels)).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        tot += loss.item() * batch.num_graphs
    train_loss = tot / len(train_loader.dataset)
    val_loss, cwa, swa, cpx, _, _ = evaluate(model, dev_loader, spr["dev"]["sequence"])
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CpxWA={cpx:.3f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cpx": cpx}
    )

# --------------------------- test evaluation ---------------------------
test_loss, cwa_t, swa_t, cpx_t, pred_lbl, true_lbl = evaluate(
    model, test_loader, spr["test"]["sequence"]
)
print(
    f"TEST:  loss={test_loss:.4f}  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}"
)
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cpx": cpx_t,
}
experiment_data["SPR_BENCH"]["predictions"] = pred_lbl
experiment_data["SPR_BENCH"]["ground_truth"] = true_lbl

# --------------------------- save ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
