import os, random, string, pathlib, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ----------------- WORKING DIR & DEVICE -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- METRIC HELPERS -----------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(1e-6, sum(w))


# ----------------- LOAD / FALLBACK DATA -----------------
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


def generate_synth(n):
    shapes = list(string.ascii_uppercase[:5])
    colors = list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(random.randint(0, 2))
    return seqs, labels


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    dset = load_spr_bench(data_root)
    print("Loaded SPR_BENCH dataset")
except Exception as e:
    print("Dataset not found, generating synthetic data:", e)
    tr_seq, tr_y = generate_synth(800)
    dv_seq, dv_y = generate_synth(200)
    ts_seq, ts_y = generate_synth(200)
    empty = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    dset = DatasetDict(
        {
            "train": empty.add_column("sequence", tr_seq).add_column("label", tr_y),
            "dev": empty.add_column("sequence", dv_seq).add_column("label", dv_y),
            "test": empty.add_column("sequence", ts_seq).add_column("label", ts_y),
        }
    )


# ----------------- VOCAB BUILDING -----------------------
def build_vocabs(split):
    shapes, colors, labels = set(), set(), set()
    for ex in split:
        for tok in ex["sequence"].split():
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
        {l: i for i, l in enumerate(sorted(labels))},
    )


shape2idx, color2idx, label2idx = build_vocabs(dset["train"])
num_shapes, len_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)


# ----------------- GRAPH CONVERSION ---------------------
def seq_to_graph(seq, label, max_pos=30) -> Data:
    toks = seq.split()
    n = len(toks)
    sh_idx = [shape2idx[t[0]] for t in toks]
    co_idx = [color2idx[t[1:]] for t in toks]
    pos_idx = [min(i, max_pos - 1) for i in range(n)]
    x = torch.tensor(list(zip(sh_idx, co_idx, pos_idx)), dtype=torch.long)
    src, dst = [], []
    # sequential edges
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    # same-shape and same-color fully connected edges
    by_shape = {}
    by_color = {}
    for i, t in enumerate(toks):
        by_shape.setdefault(sh_idx[i], []).append(i)
        by_color.setdefault(co_idx[i], []).append(i)

    def connect(groups):
        for nodes in groups.values():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    src.extend([nodes[i], nodes[j]])
                    dst.extend([nodes[j], nodes[i]])

    connect(by_shape)
    connect(by_color)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


# Build graphs (subsample train for speed)
train_graphs = [
    seq_to_graph(ex["sequence"], ex["label"])
    for ex in random.sample(list(dset["train"]), min(6000, len(dset["train"])))
]
dev_graphs = [
    seq_to_graph(ex["sequence"], ex["label"])
    for ex in random.sample(list(dset["dev"]), min(1200, len(dset["dev"])))
]
test_graphs = [
    seq_to_graph(ex["sequence"], ex["label"])
    for ex in random.sample(list(dset["test"]), min(2000, len(dset["test"])))
]


# ----------------- MODEL -------------------------------
class SPRGraphNet(nn.Module):
    def __init__(
        self, num_shapes, num_colors, num_pos, num_classes, emb_dim=16, hid=32
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, emb_dim)
        self.color_emb = nn.Embedding(num_colors, emb_dim)
        self.pos_emb = nn.Embedding(num_pos, emb_dim)
        self.gnn1 = SAGEConv(emb_dim * 3, hid)
        self.gnn2 = SAGEConv(hid, hid)
        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, data):
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        pos = self.pos_emb(data.x[:, 2])
        h = torch.cat([shp, col, pos], dim=-1)
        h = self.gnn1(h, data.edge_index).relu()
        h = self.gnn2(h, data.edge_index).relu()
        hg = global_mean_pool(h, data.batch)
        return self.classifier(hg)


# ----------------- TRAINING LOOP ------------------------
batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size * 2)
test_loader = DataLoader(test_graphs, batch_size=batch_size * 2)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "CompWA": []},
        "predictions": [],
        "ground_truth": [],
    }
}

model = SPRGraphNet(num_shapes, len_colors, 30, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
best_cwa = 0
patience = 3
wait = 0

max_epochs = 15
for epoch in range(1, max_epochs + 1):
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
    # ---- validate ----
    model.eval()
    tot_vloss = 0
    ys = []
    preds = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            tot_vloss += criterion(out, batch.y).item() * batch.num_graphs
            pred = out.argmax(dim=-1).cpu().tolist()
            ys.extend(batch.y.cpu().tolist())
            preds.extend(pred)
            seqs.extend(batch.seq)
    val_loss = tot_vloss / len(dev_loader.dataset)
    compwa = complexity_weighted_accuracy(seqs, ys, preds)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}   CompWA = {compwa:.4f}")
    # ---- bookkeeping ----
    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["CompWA"].append(compwa)
    # ---- early stopping ----
    if compwa > best_cwa:
        best_cwa = compwa
        wait = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping triggered.")
        break

# ----------------- TEST EVALUATION ----------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best_model.pt")))
model.eval()
ys = []
preds = []
seqs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds.extend(out.argmax(dim=-1).cpu().tolist())
        ys.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
test_compwa = complexity_weighted_accuracy(seqs, ys, preds)
print(f"Test CompWA = {test_compwa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = ys

# ----------------- SAVE RESULTS -------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", working_dir)
