# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, string, warnings, math, time, pathlib, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---------------- working dir & device ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- metric helpers ----------------------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ---------------- dataset utils -----------------------
def try_load_benchmark():
    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None
    try:
        from datasets import load_dataset, DatasetDict
    except ImportError:
        warnings.warn("datasets lib not installed; falling back to synthetic data")
        return None

    def _ld(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    d = {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    return d


def make_synthetic(n_tr=500, n_dev=150, n_te=200):
    shapes, colors = list(string.ascii_uppercase[:8]), list(string.ascii_lowercase[:8])

    def gen(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(4, 15)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            label = int(
                sum(t[0] == toks[0][0] for t in toks) > L / 2
            )  # simple majority rule
            seqs.append(" ".join(toks))
            labels.append(label)
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(gen(n_tr)),
            "dev": Dataset.from_dict(gen(n_dev)),
            "test": Dataset.from_dict(gen(n_te)),
        }
    )


dataset = try_load_benchmark() or make_synthetic()
print("Dataset sizes:", {k: len(v) for k, v in dataset.items()})

# ---------------- vocabularies ------------------------
shape2idx, color2idx = {}, {}


def add_tok(tok):
    s, c = tok[0], tok[1]
    if s not in shape2idx:
        shape2idx[s] = len(shape2idx)
    if c not in color2idx:
        color2idx[c] = len(color2idx)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_tok(tok)
n_shapes, n_colors = len(shape2idx), len(color2idx)
num_classes = len(set(dataset["train"]["label"]))
print(f"Shapes={n_shapes} Colors={n_colors} Classes={num_classes}")


# ---------------- seq -> graph ------------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    s_ids = [shape2idx[t[0]] for t in toks]
    c_ids = [color2idx[t[1]] for t in toks]
    x = torch.tensor(list(zip(s_ids, c_ids)), dtype=torch.long)

    src, dst, etype = [], [], []  # edge type: 0 consecutive, 1 same-shape, 2 same-color
    # consecutive
    for i in range(len(toks) - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [0, 0]
    # same-shape
    buckets = {}
    for i, s in enumerate(s_ids):
        buckets.setdefault(s, []).append(i)
    for nodes in buckets.values():
        for i in nodes:
            for j in nodes:
                if i < j:
                    src += [i, j]
                    dst += [j, i]
                    etype += [1, 1]
    # same-color
    buckets = {}
    for i, c in enumerate(c_ids):
        buckets.setdefault(c, []).append(i)
    for nodes in buckets.values():
        for i in nodes:
            for j in nodes:
                if i < j:
                    src += [i, j]
                    dst += [j, i]
                    etype += [2, 2]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    data = Data(
        x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([label])
    )
    data.seq_raw = seq
    return data


train_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["train"]["sequence"], dataset["train"]["label"])
]
dev_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["dev"]["sequence"], dataset["dev"]["label"])
]
test_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["test"]["sequence"], dataset["test"]["label"])
]


# ---------------- model --------------------------------
class RGCNClassifier(nn.Module):
    def __init__(self, n_shapes, n_colors, n_relations, emb=32, hid=64, n_cls=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb)
        self.color_emb = nn.Embedding(n_colors, emb)
        self.lin_in = nn.Linear(emb * 2, hid)
        self.conv1 = RGCNConv(hid, hid, num_relations=n_relations)
        self.conv2 = RGCNConv(hid, hid, num_relations=n_relations)
        self.out = nn.Linear(hid, n_cls)

    def forward(self, data):
        sh = self.shape_emb(data.x[:, 0])
        co = self.color_emb(data.x[:, 1])
        x = F.relu(self.lin_in(torch.cat([sh, co], dim=-1)))
        x = F.relu(self.conv1(x, data.edge_index, data.edge_type))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_type))
        x = global_mean_pool(x, data.batch)
        return self.out(x)


# --------------- experiment logger --------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# --------------- helpers ------------------------------
def build_loader(graphs, bs, shuffle=False):
    return DataLoader(graphs, batch_size=bs, shuffle=shuffle)


train_loader = build_loader(train_graphs, 32, True)
dev_loader = build_loader(dev_graphs, 64)
test_loader = build_loader(test_graphs, 64)

# class weights
labels_tensor = torch.tensor(dataset["train"]["label"])
class_weights = torch.bincount(labels_tensor).float()
class_weights = 1.0 / (class_weights + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(device)

# ---------------- training -----------------------------
model = RGCNClassifier(n_shapes, n_colors, n_relations=3, n_cls=num_classes).to(device)
optimizer = Adam(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss(weight=class_weights)

best_hpa, best_state = -1, None
EPOCHS = 15

for epoch in range(1, EPOCHS + 1):
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
    train_loss = tot_loss / len(train_loader.dataset)
    experiment_data["SPR"]["losses"]["train"].append(train_loss)

    # validate
    model.eval()
    v_loss, preds, gts, seqs = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            v_loss += loss.item() * batch.num_graphs
            p = logits.argmax(-1).cpu().tolist()
            g = batch.y.cpu().tolist()
            s = batch.seq_raw
            preds.extend(p)
            gts.extend(g)
            seqs.extend(s)
    v_loss /= len(dev_loader.dataset)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hpa = harmonic_poly_accuracy(cwa, swa)

    print(
        f"Epoch {epoch}: validation_loss = {v_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}"
    )

    experiment_data["SPR"]["losses"]["val"].append(v_loss)
    experiment_data["SPR"]["metrics"]["val"].append(
        {"CWA": cwa, "SWA": swa, "HPA": hpa}
    )
    experiment_data["SPR"]["epochs"].append(epoch)

    if hpa > best_hpa:
        best_hpa = hpa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    scheduler.step()

# ---------------- test with best model ----------------
model.load_state_dict(best_state)
model.to(device)
model.eval()
preds, gts, seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq_raw)
cwa = color_weighted_accuracy(seqs, gts, preds)
swa = shape_weighted_accuracy(seqs, gts, preds)
hpa = harmonic_poly_accuracy(cwa, swa)
print(f"TEST  | CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}")

experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved logs ->", os.path.join(working_dir, "experiment_data.npy"))
