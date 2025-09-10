import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():  # tiny synthetic fallback
    print("SPR_BENCH not found – creating tiny synthetic data.")
    os.makedirs(DATA_PATH, exist_ok=True)
    for split, size in [("train", 200), ("dev", 40), ("test", 40)]:
        rng = np.random.default_rng(0)
        shapes, colours = ["A", "B", "C"], ["1", "2", "3"]
        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            import csv

            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(size):
                n = rng.integers(3, 7)
                seq = " ".join(
                    rng.choice(shapes) + rng.choice(colours) for _ in range(n)
                )
                w.writerow([f"{split}_{i}", seq, rng.choice(["yes", "no"])])

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------- vocab ----------
def parse_token(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


shapes, colours = set(), set()
for r in dsets["train"]:
    for tok in r["sequence"].split():
        s, c = parse_token(tok)
        shapes.add(s)
        colours.add(c)
shape2id = {s: i for i, s in enumerate(sorted(shapes))}
col2id = {c: i for i, c in enumerate(sorted(colours))}
print("Shapes:", shape2id)
print("Colours:", col2id)

label2id = {l: i for i, l in enumerate(sorted({r["label"] for r in dsets["train"]}))}


# ---------- shape-only seq→graph ----------
def seq_to_graph(sequence, lbl):
    toks = sequence.split()
    n = len(toks)
    x = []
    for tok in toks:
        s, _ = parse_token(tok)
        vec = np.zeros(len(shape2id) + len(col2id), dtype=np.float32)
        vec[shape2id[s]] = 1.0  # shape bit only
        # colour bits deliberately left zero (shape-only ablation)
        x.append(vec)
    x = torch.tensor(np.stack(x))
    if n > 1:
        src = torch.arange(0, n - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train = build_graph_dataset("train")
graph_dev = build_graph_dataset("dev")
graph_test = build_graph_dataset("test")

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- metrics ----------
def complexity_weight(seq):
    toks = seq.split()
    return len({t[0] for t in toks}) + len({t[1:] if len(t) > 1 else "0" for t in toks})


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


# ---------- experiment_data dict ----------
experiment_data = {
    "shape_only": {
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
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch.num_graphs
        tcorr += int((out.argmax(-1) == batch.y).sum().item())
        tex += batch.num_graphs
    tr_loss = tloss / tex
    tr_acc = tcorr / tex
    experiment_data["shape_only"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["shape_only"]["SPR_BENCH"]["metrics"]["train"].append(tr_acc)

    # ---- validation ----
    model.eval()
    vloss = vcorr = vex = 0
    all_pred = all_gt = all_seq = []
    with torch.no_grad():
        for batch, raw in zip(dev_loader, dsets["dev"]):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            vloss += loss.item() * batch.num_graphs
            pred = out.argmax(-1).cpu()
            vcorr += int((pred == batch.y.cpu()).sum().item())
            vex += batch.num_graphs
            all_pred.extend(pred.tolist())
            all_gt.extend(batch.y.cpu().tolist())
            all_seq.append(raw["sequence"])
    val_loss = vloss / vex
    val_acc = vcorr / vex
    experiment_data["shape_only"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["shape_only"]["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

# ---------- Comp-Weighted Accuracy on dev ----------
seqs = [r["sequence"] for r in dsets["dev"]]
model.eval()
preds = []
with torch.no_grad():
    for batch in dev_loader:
        batch = batch.to(device)
        preds.extend(
            model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
        )
compwa = comp_weighted_accuracy(
    seqs, [label2id[r["label"]] for r in dsets["dev"]], preds
)
print(f"Complexity-Weighted Accuracy (dev): {compwa:.4f}")

experiment_data["shape_only"]["SPR_BENCH"]["predictions"] = preds
experiment_data["shape_only"]["SPR_BENCH"]["ground_truth"] = [
    label2id[r["label"]] for r in dsets["dev"]
]

# ---------- save & plot ----------
plt.figure()
plt.plot(experiment_data["shape_only"]["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["shape_only"]["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Cross-Entropy loss (shape-only ablation)")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Data & plot saved to ./working")
