import os, pathlib, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from typing import List, Dict
from datasets import load_dataset

# ---------- work dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helpers -----------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / (sum(weights) if sum(weights) else 1)


# ---------- load data ----------
spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))


def _load_csv(name):  # helper
    return load_dataset(
        "csv",
        data_files=str(spr_root / f"{name}.csv"),
        split="train",
        cache_dir=".cache_dsets",
    )


if spr_root.exists():
    dsets = {split: _load_csv(split) for split in ["train", "dev", "test"]}
else:
    print("!! SPR_BENCH not found – creating tiny synthetic set for demo.")

    def synth(n):
        shapes, colors = list("ABC"), list("123")
        seqs, labels = [], []
        for _ in range(n):
            L = np.random.randint(4, 9)
            seqs.append(
                " ".join(
                    np.random.choice(shapes) + np.random.choice(colors)
                    for _ in range(L)
                )
            )
            labels.append(np.random.randint(0, 3))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    dsets = {"train": synth(600), "dev": synth(200), "test": synth(200)}

# ---------- vocab ----------
all_shapes, all_colors = set(), set()
for s in dsets["train"]["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0])
        all_colors.add(tok[1])
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
num_shapes, num_colors = len(shape2idx), len(color2idx)
num_classes = len(set(dsets["train"]["label"]))


# ---------- graph builder -----
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.strip().split()
    n = len(toks)
    shp_idx = [shape2idx[t[0]] for t in toks]
    col_idx = [color2idx[t[1]] for t in toks]
    x = torch.tensor(np.stack([shp_idx, col_idx], 1), dtype=torch.long)

    # relation 0: adjacency
    src_adj = list(range(n - 1))
    dst_adj = list(range(1, n))
    edges_0 = [(s, d) for s, d in zip(src_adj, dst_adj)] + [
        (d, s) for s, d in zip(src_adj, dst_adj)
    ]
    # relation 1: same color
    edges_1 = []
    for c in set(col_idx):
        inds = [i for i, ci in enumerate(col_idx) if ci == c]
        edges_1 += [(i, j) for i in inds for j in inds if i != j]
    # relation 2: same shape
    edges_2 = []
    for sh in set(shp_idx):
        inds = [i for i, si in enumerate(shp_idx) if si == sh]
        edges_2 += [(i, j) for i in inds for j in inds if i != j]

    all_edges, rel_types = [], []
    for rel, e_list in enumerate([edges_0, edges_1, edges_2]):
        all_edges += e_list
        rel_types += [rel] * len(e_list)

    if len(all_edges) == 0:  # single-node graph fallback
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor(np.array(all_edges).T, dtype=torch.long)
        edge_type = torch.tensor(rel_types, dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


def build_dataset(split) -> List[Data]:
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_data, dev_data, test_data = map(
    build_dataset, (dsets["train"], dsets["dev"], dsets["test"])
)


# ---------- model --------------
class SPR_RGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, 8)
        self.color_emb = nn.Embedding(num_colors, 8)
        self.pre_lin = nn.Linear(16, 32)
        self.rgcn1 = RGCNConv(32, 64, num_relations=3)
        self.rgcn2 = RGCNConv(64, 64, num_relations=3)
        self.cls = nn.Linear(64, num_classes)

    def forward(self, data: Data):
        x = torch.cat(
            [self.shape_emb(data.x[:, 0]), self.color_emb(data.x[:, 1])], dim=1
        )
        x = F.relu(self.pre_lin(x))
        x = F.relu(self.rgcn1(x, data.edge_index, data.edge_type))
        x = F.relu(self.rgcn2(x, data.edge_index, data.edge_type))
        x = global_mean_pool(x, data.batch)
        return self.cls(x)


# ---------- experiment storage ---
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_cplxwa": [], "val_cplxwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- training -------------
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=256, shuffle=False)
model = SPR_RGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 8
for epoch in range(1, epochs + 1):
    # train
    model.train()
    tot_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
    avg_train_loss = tot_loss / len(train_loader.dataset)

    # evaluate on train set for metric
    model.eval()
    with torch.no_grad():
        all_seq_t, all_true_t, all_pred_t = [], [], []
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch).argmax(1).cpu().tolist()
            tru = batch.y.cpu().tolist()
            all_pred_t.extend(pred)
            all_true_t.extend(tru)
            all_seq_t.extend(batch.seq)
        train_cplx = complexity_weighted_accuracy(all_seq_t, all_true_t, all_pred_t)

    # dev
    val_loss = 0
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            val_loss += criterion(out, batch.y).item() * batch.num_graphs
            all_pred.extend(out.argmax(1).cpu().tolist())
            all_true.extend(batch.y.cpu().tolist())
            all_seq.extend(batch.seq)
    avg_val_loss = val_loss / len(dev_loader.dataset)
    val_cplx = complexity_weighted_accuracy(all_seq, all_true, all_pred)

    # store
    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(avg_train_loss)
    ed["losses"]["val"].append(avg_val_loss)
    ed["metrics"]["train_cplxwa"].append(train_cplx)
    ed["metrics"]["val_cplxwa"].append(val_cplx)
    ed["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
        f"train_CplxWA={train_cplx:.4f} val_CplxWA={val_cplx:.4f}"
    )

# ---------- test -----------------
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
model.eval()
all_seq, all_true, all_pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pr = model(batch).argmax(1).cpu().tolist()
        all_pred.extend(pr)
        all_true.extend(batch.y.cpu().tolist())
        all_seq.extend(batch.seq)
test_cplx = complexity_weighted_accuracy(all_seq, all_true, all_pred)
experiment_data["SPR_BENCH"]["predictions"] = all_pred
experiment_data["SPR_BENCH"]["ground_truth"] = all_true
experiment_data["SPR_BENCH"]["metrics"]["test_cplxwa"] = test_cplx
print(f"Test Complexity-Weighted Accuracy: {test_cplx:.4f}")

# ---------- save -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
