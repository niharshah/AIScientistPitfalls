import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():
    print("SPR_BENCH not found – creating tiny synthetic data.")
    os.makedirs(DATA_PATH, exist_ok=True)
    rng = np.random.default_rng(0)
    shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
    for split, size in [("train", 200), ("dev", 40), ("test", 40)]:
        with open(DATA_PATH / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(size):
                n = rng.integers(3, 7)
                seq = " ".join(
                    rng.choice(shapes) + rng.choice(colors) for _ in range(n)
                )
                lbl = rng.choice(["yes", "no"])
                f.write(f"{split}_{i},{seq},{lbl}\n")

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------- vocab ----------
def parse_token(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


shapes, colours = set(), set()
for row in dsets["train"]:
    for tok in row["sequence"].split():
        s, c = parse_token(tok)
        shapes.add(s)
        colours.add(c)
shape2id = {s: i for i, s in enumerate(sorted(shapes))}
col2id = {c: i for i, c in enumerate(sorted(colours))}
label2id = {l: i for i, l in enumerate(sorted({r["label"] for r in dsets["train"]}))}
print("Shapes:", shape2id, "Colours:", col2id)


# ---------- fully-connected graph builder ----------
def seq_to_graph_fc(sequence, lbl):
    toks = sequence.split()
    n = len(toks)
    feats = []
    for t in toks:
        s, c = parse_token(t)
        vec = np.zeros(len(shape2id) + len(col2id), dtype=np.float32)
        vec[shape2id[s]] = 1.0
        vec[len(shape2id) + col2id[c]] = 1.0
        feats.append(vec)
    x = torch.tensor(np.stack(feats))
    if n > 1:
        idx = torch.arange(n, dtype=torch.long)
        pairs = torch.combinations(idx, r=2)
        edge_index = torch.cat([pairs.t(), pairs.flip(1).t()], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph_fc(r["sequence"], r["label"]) for r in dsets[split]]


graph_train, graph_dev, graph_test = map(build_graph_dataset, ["train", "dev", "test"])
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)
test_loader = DataLoader(graph_test, batch_size=128, shuffle=False)


# ---------- model ----------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64, num_classes=len(label2id)):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = GCN(len(shape2id) + len(col2id)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- complexity-weighted accuracy ----------
def complexity_weight(seq):
    ts = seq.split()
    return len({t[0] for t in ts}) + len({t[1:] if len(t) > 1 else "0" for t in ts})


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    return sum(wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)) / sum(w)


# ---------- experiment data ----------
experiment_data = {
    "FullyConnected": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- training ----------
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.train()
    tloss = tcorr = tex = 0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        opt.step()
        tloss += loss.item() * batch.num_graphs
        tcorr += int((out.argmax(-1) == batch.y).sum())
        tex += batch.num_graphs
    experiment_data["FullyConnected"]["SPR_BENCH"]["losses"]["train"].append(
        tloss / tex
    )
    experiment_data["FullyConnected"]["SPR_BENCH"]["metrics"]["train"].append(
        tcorr / tex
    )

    # validation
    model.eval()
    vloss = vcorr = vex = 0
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            vloss += loss.item() * batch.num_graphs
            vcorr += int((out.argmax(-1) == batch.y).sum())
            vex += batch.num_graphs
    experiment_data["FullyConnected"]["SPR_BENCH"]["losses"]["val"].append(vloss / vex)
    experiment_data["FullyConnected"]["SPR_BENCH"]["metrics"]["val"].append(vcorr / vex)
    print(f"Epoch {epoch}: val_loss={vloss/vex:.4f}, val_acc={vcorr/vex:.4f}")

# ---------- final eval on dev ----------
seqs = [r["sequence"] for r in dsets["dev"]]
model.eval()
preds = []
with torch.no_grad():
    for batch in dev_loader:
        batch = batch.to(device)
        preds.extend(
            model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
        )
gt = [label2id[r["label"]] for r in dsets["dev"]]
compwa = comp_weighted_accuracy(seqs, gt, preds)
print("Complexity-Weighted Accuracy (dev):", compwa)
experiment_data["FullyConnected"]["SPR_BENCH"]["predictions"] = preds
experiment_data["FullyConnected"]["SPR_BENCH"]["ground_truth"] = gt

# ---------- plot & save ----------
plt.figure()
plt.plot(
    experiment_data["FullyConnected"]["SPR_BENCH"]["losses"]["train"], label="train"
)
plt.plot(experiment_data["FullyConnected"]["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Cross-Entropy loss (Fully-Connected)")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved data & plot to ./working")
