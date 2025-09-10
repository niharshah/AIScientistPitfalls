import os, pathlib, time, numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, LayerNorm

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- dataset loader ----------------
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
if not DATA_PATH.exists():
    # tiny synthetic back-up so script is runnable
    print("SPR_BENCH not found – writing tiny synthetic data.")
    DATA_PATH.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in [("train", 300), ("dev", 60), ("test", 60)]:
        with open(DATA_PATH / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                L = rng.integers(3, 8)
                seq = " ".join(
                    rng.choice(list("ABC")) + str(rng.integers(1, 4)) for _ in range(L)
                )
                label = rng.choice(["yes", "no"])
                f.write(f"{split}_{i},{seq},{label}\n")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------------- vocab ----------------
def token_parts(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


shapes, colors = set(), set()
for ex in dsets["train"]:
    for t in ex["sequence"].split():
        s, c = token_parts(t)
        shapes.add(s)
        colors.add(c)
shape2id = {s: i for i, s in enumerate(sorted(shapes))}
color2id = {c: i for i, c in enumerate(sorted(colors))}
label2id = {l: i for i, l in enumerate(sorted({ex["label"] for ex in dsets["train"]}))}


# ---------------- utils for metrics ----------------
def count_color_variety(seq):
    return len({t[1:] if len(t) > 1 else "0" for t in seq.split()})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split()})


def complexity_weight(seq):
    return count_color_variety(seq) * count_shape_variety(seq)


def weighted_accuracy(weights, ytrue, ypred):
    corr = [w if a == b else 0 for w, a, b in zip(weights, ytrue, ypred)]
    return sum(corr) / sum(weights) if weights else 0.0


# ---------------- graph construction ----------------
def sequence_to_graph(sequence, label):
    toks = sequence.split()
    n = len(toks)
    # node features
    feats = []
    for t in toks:
        s, c = token_parts(t)
        v = np.zeros(len(shape2id) + len(color2id), dtype=np.float32)
        v[shape2id[s]] = 1.0
        v[len(shape2id) + color2id[c]] = 1.0
        feats.append(v)
    x = torch.tensor(np.stack(feats))
    # edges: sequential + same shape + same color
    src, dst = [], []
    # sequential
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    # same shape/color
    by_shape, by_color = {}, {}
    for i, t in enumerate(toks):
        s, c = token_parts(t)
        by_shape.setdefault(s, []).append(i)
        by_color.setdefault(c, []).append(i)
    for group in list(by_shape.values()) + list(by_color.values()):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                src += [group[i], group[j]]
                dst += [group[j], group[i]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2id[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=sequence)


def build_graph_dataset(split):
    return [sequence_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


train_graphs = build_graph_dataset("train")
dev_graphs = build_graph_dataset("dev")
test_graphs = build_graph_dataset("test")

train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=256, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=256, shuffle=False)


# ---------------- model ----------------
class GraphModel(torch.nn.Module):
    def __init__(self, in_dim, hid=64, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.ln1 = LayerNorm(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.ln2 = LayerNorm(hid)
        self.lin = torch.nn.Linear(hid, num_classes)
        self.drop = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = self.drop(self.ln1(self.conv1(x, edge_index).relu()))
        x = self.drop(self.ln2(self.conv2(x, edge_index).relu()))
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = GraphModel(len(shape2id) + len(color2id), hid=96, num_classes=len(label2id)).to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------- tracking dict ----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "train_cxa": [], "val_cxa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- training loop ----------------
EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tot_loss = tot_corr = tot_graphs = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(-1)
        tot_loss += loss.item() * batch.num_graphs
        tot_corr += int((pred == batch.y).sum())
        tot_graphs += batch.num_graphs
    tr_loss = tot_loss / tot_graphs
    tr_acc = tot_corr / tot_graphs
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)

    # ---- validation ----
    model.eval()
    v_loss = v_corr = v_graphs = 0
    all_pred, all_gt, all_seq = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            pred = out.argmax(-1).cpu()
            v_loss += loss.item() * batch.num_graphs
            v_corr += int((pred == batch.y.cpu()).sum())
            v_graphs += batch.num_graphs
            all_pred.extend(pred.tolist())
            all_gt.extend(batch.y.cpu().tolist())
            all_seq.extend(batch.seq)
    val_loss = v_loss / v_graphs
    val_acc = v_corr / v_graphs
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)

    # ---- CXA ----
    tr_cxa = 0
    with torch.no_grad():
        for g in train_graphs:
            out = model(
                g.x.to(device),
                g.edge_index.to(device),
                torch.zeros(g.x.size(0), dtype=torch.long, device=device),
            )
            tr_cxa += (
                complexity_weight(g.seq) if out.argmax().item() == g.y.item() else 0
            )
    tr_cxa /= sum(complexity_weight(g.seq) for g in train_graphs)
    val_cxa = weighted_accuracy(
        [complexity_weight(s) for s in all_seq], all_gt, all_pred
    )
    experiment_data["SPR_BENCH"]["metrics"]["train_cxa"].append(tr_cxa)
    experiment_data["SPR_BENCH"]["metrics"]["val_cxa"].append(val_cxa)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.3f} | val_CXA={val_cxa:.3f}"
    )

# ---------------- test evaluation ----------------
model.eval()
test_pred, test_gt, test_seq = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        p = out.argmax(-1).cpu()
        test_pred.extend(p.tolist())
        test_gt.extend(batch.y.cpu().tolist())
        test_seq.extend(batch.seq)
CWA = weighted_accuracy([count_color_variety(s) for s in test_seq], test_gt, test_pred)
SWA = weighted_accuracy([count_shape_variety(s) for s in test_seq], test_gt, test_pred)
CXA = weighted_accuracy([complexity_weight(s) for s in test_seq], test_gt, test_pred)
print(f"Test  CWA={CWA:.3f}  SWA={SWA:.3f}  CXA={CXA:.3f}")

# save overall data
experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to ./working")
