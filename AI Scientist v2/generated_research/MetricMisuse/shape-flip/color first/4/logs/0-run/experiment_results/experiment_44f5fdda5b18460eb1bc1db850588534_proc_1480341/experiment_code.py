import os, random, string, pathlib, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# working directory & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# helper metrics ----------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) + 1e-9)


# ------------------------------------------------------------------
# dataset loading (real path or fallback synthetic) -----------------
BENCH_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def gen_synth(n):
    shapes = list(string.ascii_uppercase[:5])
    colors = list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(random.randint(0, 2))
    return seqs, labels


try:
    spr = load_spr_bench(BENCH_PATH)
    print("Loaded SPR_BENCH from disk.")
except Exception as e:
    print("Dataset not found, generating synthetic toy data.", e)
    tr_s, tr_y = gen_synth(5000)
    dv_s, dv_y = gen_synth(1000)
    ts_s, ts_y = gen_synth(1000)
    empty = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    spr = DatasetDict(
        {
            "train": empty.add_column("sequence", tr_s).add_column("label", tr_y),
            "dev": empty.add_column("sequence", dv_s).add_column("label", dv_y),
            "test": empty.add_column("sequence", ts_s).add_column("label", ts_y),
        }
    )

# optional subsampling for speed -------------------------------------------------
MAX_TRAIN, MAX_DEV, MAX_TEST = 5000, 1000, 1000
spr["train"] = (
    spr["train"].shuffle(seed=42).select(range(min(len(spr["train"]), MAX_TRAIN)))
)
spr["dev"] = spr["dev"].shuffle(seed=42).select(range(min(len(spr["dev"]), MAX_DEV)))
spr["test"] = (
    spr["test"].shuffle(seed=42).select(range(min(len(spr["test"]), MAX_TEST)))
)


# ------------------------------------------------------------------
# vocabularies ------------------------------------------------------
def build_vocabs(dataset):
    shapes, colors, labels = set(), set(), set()
    for ex in dataset:
        for tok in ex["sequence"].split():
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
        {l: i for i, l in enumerate(sorted(labels))},
    )


shape2idx, color2idx, label2idx = build_vocabs(spr["train"])
num_shapes, num_colors, num_labels = len(shape2idx), len(color2idx), len(label2idx)


# ------------------------------------------------------------------
# graph conversion -------------------------------------------------
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shape_idx = [shape2idx[t[0]] for t in toks]
    color_idx = [color2idx[t[1:]] for t in toks]
    # node features
    x = torch.tensor(list(zip(shape_idx, color_idx)), dtype=torch.long)
    edge_src, edge_dst, edge_type = [], [], []
    # 0 = sequential relation
    for i in range(n - 1):
        edge_src += [i, i + 1]
        edge_dst += [i + 1, i]
        edge_type += [0, 0]
    # 1 = same shape, 2 = same color
    shape_groups = {}
    color_groups = {}
    for idx, (s, c) in enumerate(zip(shape_idx, color_idx)):
        shape_groups.setdefault(s, []).append(idx)
        color_groups.setdefault(c, []).append(idx)
    for nodes, r in [(shape_groups, 1), (color_groups, 2)]:
        for verts in nodes.values():
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    edge_src += [verts[i], verts[j]]
                    edge_dst += [verts[j], verts[i]]
                    edge_type += [r, r]
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["test"]]


# ------------------------------------------------------------------
# model ------------------------------------------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, n_shapes, n_colors, n_classes, emb_dim=16, hidden=32, num_rel=3):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb_dim)
        self.color_emb = nn.Embedding(n_colors, emb_dim)
        in_dim = emb_dim * 2
        self.conv1 = RGCNConv(in_dim, hidden, num_rel)
        self.conv2 = RGCNConv(hidden, hidden, num_rel)
        self.cls = nn.Linear(hidden, n_classes)

    def forward(self, data):
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        h = torch.cat([shp, col], dim=-1)
        h = self.conv1(h, data.edge_index, data.edge_type).relu()
        h = self.conv2(h, data.edge_index, data.edge_type).relu()
        g = global_mean_pool(h, data.batch)
        return self.cls(g)


# ------------------------------------------------------------------
# data loaders ------------------------------------------------------
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128)
test_loader = DataLoader(test_graphs, batch_size=128)

# ------------------------------------------------------------------
# experiment data structure ----------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------
# training ---------------------------------------------------------
model = SPR_RGCN(num_shapes, num_colors, num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 10
t0 = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    # ---- train ----
    model.train()
    tot_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
    train_loss = tot_loss / len(train_loader.dataset)
    # ---- val ----
    model.eval()
    vloss, ys, preds, seqs = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            vloss += criterion(out, batch.y).item() * batch.num_graphs
            pred = out.argmax(dim=-1).cpu().tolist()
            ys.extend(batch.y.cpu().tolist())
            preds.extend(pred)
            seqs.extend(batch.seq)
    val_loss = vloss / len(dev_loader.dataset)
    val_comp = complexity_weighted_accuracy(seqs, ys, preds)
    # logging
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  CompWA = {val_comp:.4f}")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_comp)

# ------------------------------------------------------------------
# final test evaluation --------------------------------------------
model.eval()
ys, preds, seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds.extend(out.argmax(dim=-1).cpu().tolist())
        ys.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
test_comp = complexity_weighted_accuracy(seqs, ys, preds)
print(f"Test Complexity-Weighted Accuracy: {test_comp:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = ys
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy ; total runtime %.1fs" % (time.time() - t0))
