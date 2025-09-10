import os, pathlib, random, itertools, time, numpy as np, torch, torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from collections import defaultdict
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---------------- working dir & experiment data ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- utility funcs ----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / (sum(w) or 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / (sum(w) or 1)


# ---------------- data preparation ----------------
def prepare_graphs(spr):
    shapes = set()
    colors = set()
    for seq in spr["train"]["sequence"]:
        for tok in seq.split():
            shapes.add(tok[0])
            colors.add(tok[1])
    shape2i = {s: i for i, s in enumerate(sorted(shapes))}
    color2i = {c: i for i, c in enumerate(sorted(colors))}
    labels = sorted(set(spr["train"]["label"]))
    lab2i = {l: i for i, l in enumerate(labels)}

    def seq_to_graph(seq, label):
        toks = seq.split()
        n = len(toks)
        shape_idx = torch.tensor([shape2i[t[0]] for t in toks], dtype=torch.long)
        color_idx = torch.tensor([color2i[t[1]] for t in toks], dtype=torch.long)
        pos_idx = torch.tensor(list(range(n)), dtype=torch.long)

        edges, etypes = set(), []
        # adjacency edges (rel 0)
        for i in range(n - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))
        # same color (rel 1) and same shape (rel 2)
        by_color = defaultdict(list)
        by_shape = defaultdict(list)
        for i, t in enumerate(toks):
            by_color[t[1]].append(i)
            by_shape[t[0]].append(i)
        for lst in by_color.values():
            for i in lst:
                for j in lst:
                    if i != j:
                        edges.add((i, j))
        for lst in by_shape.values():
            for i in lst:
                for j in lst:
                    if i != j:
                        edges.add((i, j))
        # build tensors
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        # build edge types parallel to edge_index
        for src, dst in edge_index.t().tolist():
            if abs(src - dst) == 1:
                etypes.append(0)
            elif toks[src][1] == toks[dst][1]:
                etypes.append(1)
            else:
                etypes.append(2)
        edge_type = torch.tensor(etypes, dtype=torch.long)
        x = torch.stack([shape_idx, color_idx, pos_idx], dim=1)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([lab2i[label]], dtype=torch.long),
        )

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = [
            seq_to_graph(s, l)
            for s, l in zip(spr[split]["sequence"], spr[split]["label"])
        ]
    return out, shape2i, color2i, lab2i


# ---------------- dataset loading ----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    from datasets import Dataset

    def synth(n):
        shapes, colors = "AB", "12"
        seqs = [
            " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 8))
            )
            for _ in range(n)
        ]
        labels = [random.choice(["yes", "no"]) for _ in range(n)]
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = DatasetDict(
        {
            "train": Dataset.from_dict(synth(2000)),
            "dev": Dataset.from_dict(synth(400)),
            "test": Dataset.from_dict(synth(400)),
        }
    )

graphs, shape2i, color2i, lab2i = prepare_graphs(spr)
inv_lab = {v: k for k, v in lab2i.items()}
max_pos = max(len(g.x) for g in graphs["train"]) + 1
num_class = len(lab2i)

train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
dev_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)


# ---------------- model ----------------
class RGCNClassifier(nn.Module):
    def __init__(self, n_shape, n_color, max_pos, emb_dim, hid_dim, n_class, num_rel=3):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(max_pos, emb_dim)
        self.conv1 = RGCNConv(emb_dim, hid_dim, num_rel)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_rel)
        self.lin = nn.Linear(hid_dim, n_class)

    def forward(self, batch):
        s, c, p = batch.x[:, 0], batch.x[:, 1], batch.x[:, 2]
        x = self.shape_emb(s) + self.color_emb(c) + self.pos_emb(p)
        x = self.conv1(x, batch.edge_index, batch.edge_type).relu()
        x = self.conv2(x, batch.edge_index, batch.edge_type).relu()
        x = global_mean_pool(x, batch.batch)
        return self.lin(x)


model = RGCNClassifier(
    len(shape2i), len(color2i), max_pos, emb_dim=32, hid_dim=64, n_class=num_class
).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
EPOCHS = 25


# ---------------- evaluation helper ----------------
def evaluate(loader, seqs):
    model.eval()
    all_p, all_t = [], []
    loss_sum = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            loss_sum += loss.item() * batch.num_graphs
            all_p.extend(out.argmax(-1).cpu().tolist())
            all_t.extend(batch.y.cpu().tolist())
    avg_loss = loss_sum / len(loader.dataset)
    pred_lbl = [inv_lab[p] for p in all_p]
    true_lbl = [inv_lab[t] for t in all_t]
    cwa = color_weighted_accuracy(seqs, true_lbl, pred_lbl)
    swa = shape_weighted_accuracy(seqs, true_lbl, pred_lbl)
    cpx = complexity_weighted_accuracy(seqs, true_lbl, pred_lbl)
    return avg_loss, cwa, swa, cpx, pred_lbl, true_lbl


# ---------------- training loop ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    train_loss = epoch_loss / len(train_loader.dataset)
    val_loss, cwa, swa, cpx, _, _ = evaluate(dev_loader, spr["dev"]["sequence"])
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CpxWA={cpx:.3f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cpxwa": cpx}
    )

# ---------------- final evaluation on test ----------------
test_loss, cwa_t, swa_t, cpx_t, preds_lbl, tru_lbl = evaluate(
    test_loader, spr["test"]["sequence"]
)
print(f"TEST  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}")
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cpxwa": cpx_t,
}
experiment_data["SPR_BENCH"]["predictions"] = preds_lbl
experiment_data["SPR_BENCH"]["ground_truth"] = tru_lbl

# ---------------- save ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
