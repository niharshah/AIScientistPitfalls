import os, pathlib, random, itertools, time, json, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ------------------- EXPERIMENT LOG --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "No-Pos": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------- DEVICE ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------- DATA HELPERS ----------------------
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


def count_color_variety(seq):  # CWA
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):  # SWA
    return len({tok[0] for tok in seq.split() if tok})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


# ------------------- GRAPH CONSTRUCTION ----------------
def build_graph_dataset(spr_dict):
    shapes = sorted(
        {
            tok[0]
            for seq in itertools.chain(*(d["sequence"] for d in spr_dict.values()))
            for tok in seq.split()
        }
    )
    colors = sorted(
        {
            tok[1]
            for seq in itertools.chain(*(d["sequence"] for d in spr_dict.values()))
            for tok in seq.split()
        }
    )
    labels = sorted({l for l in spr_dict["train"]["label"]})
    shape2i, color2i, lab2i = (
        {s: i for i, s in enumerate(shapes)},
        {c: i for i, c in enumerate(colors)},
        {l: i for i, l in enumerate(labels)},
    )

    def seq_to_graph(seq, label):
        toks = seq.split()
        n = len(toks)
        shape_idx = torch.tensor([shape2i[t[0]] for t in toks])
        color_idx = torch.tensor([color2i[t[1]] for t in toks])
        pos_idx = torch.tensor(list(range(n)))
        edge_src, edge_dst, edge_type = [], [], []
        for i in range(n - 1):
            for s, d in ((i, i + 1), (i + 1, i)):
                edge_src.append(s)
                edge_dst.append(d)
                edge_type.append(0)
        groups = defaultdict(list)
        for idx, tok in enumerate(toks):
            groups[("shape", tok[0])].append(idx)
            groups[("color", tok[1])].append(idx)
        for (k, _), idxs in groups.items():
            rel = 1 if k == "shape" else 2
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_type.append(rel)
        if not edge_src:
            edge_src, edge_dst, edge_type = [0], [0], [0]
        x = torch.stack([shape_idx, color_idx, pos_idx], 1)
        return Data(
            x=x,
            edge_index=torch.tensor([edge_src, edge_dst]),
            edge_type=torch.tensor(edge_type),
            y=torch.tensor([lab2i[label]]),
        )

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = [
            seq_to_graph(s, l)
            for s, l in zip(spr_dict[split]["sequence"], spr_dict[split]["label"])
        ]
    return out, shape2i, color2i, lab2i


# ------------------- MODEL -----------------------------
class RGCNClassifier_NoPos(nn.Module):
    """No positional embedding added (shape_emb + color_emb only)."""

    def __init__(self, n_shape, n_color, max_pos, emb_dim, hid_dim, n_class):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        # pos_emb is kept but frozen & zeroed for fairness (alternative: delete).
        self.register_buffer("zero_pos_emb", torch.zeros(1, emb_dim))
        self.conv1 = RGCNConv(emb_dim, hid_dim, num_relations=3)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_relations=3)
        self.lin = nn.Linear(hid_dim, n_class)

    def forward(self, data):
        s, c, p = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        h = self.shape_emb(s) + self.color_emb(c)  # NO positional contribution
        h = self.conv1(h, data.edge_index, data.edge_type).relu()
        h = self.conv2(h, data.edge_index, data.edge_type).relu()
        h = global_mean_pool(h, data.batch)
        return self.lin(h)


# ------------------- LOAD DATA -------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr_raw = load_spr_bench(DATA_PATH)
else:  # synthetic fallback

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

    from datasets import Dataset

    spr_raw = DatasetDict(
        {
            "train": Dataset.from_dict(synth(2000)),
            "dev": Dataset.from_dict(synth(400)),
            "test": Dataset.from_dict(synth(400)),
        }
    )

graphs, shape2i, color2i, lab2i = build_graph_dataset(spr_raw)
max_pos = max(len(g.x) for g in graphs["train"]) + 2
inv_lab = {v: k for k, v in lab2i.items()}
num_class = len(lab2i)

train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
val_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)

# ------------------- INIT MODEL ------------------------
model = RGCNClassifier_NoPos(
    len(shape2i), len(color2i), max_pos, emb_dim=32, hid_dim=64, n_class=num_class
).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
EPOCHS = 25
best_val_cpx, best_state = -1, None


# ------------------- EVAL FUNCTION ---------------------
def run_eval(loader, seqs):
    model.eval()
    tot_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b)
            loss = cross_entropy(out, b.y)
            tot_loss += loss.item() * b.num_graphs
            preds.extend(out.argmax(-1).cpu().tolist())
            trues.extend(b.y.cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    pred_lbl = [inv_lab[p] for p in preds]
    true_lbl = [inv_lab[t] for t in trues]
    cwa = color_weighted_accuracy(seqs, true_lbl, pred_lbl)
    swa = shape_weighted_accuracy(seqs, true_lbl, pred_lbl)
    cpx = complexity_weighted_accuracy(seqs, true_lbl, pred_lbl)
    return avg_loss, cwa, swa, cpx, pred_lbl, true_lbl


# ------------------- TRAIN LOOP ------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    train_loss = epoch_loss / len(train_loader.dataset)

    val_loss, cwa, swa, cpx, _, _ = run_eval(val_loader, spr_raw["dev"]["sequence"])
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CpxWA={cpx:.3f}"
    )

    experiment_data["No-Pos"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["No-Pos"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["No-Pos"]["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "cwa": cwa, "swa": swa, "cpxwa": cpx}
    )

    if cpx > best_val_cpx:
        best_val_cpx = cpx
        best_state = model.state_dict()

# ------------------- TEST ------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
t_loss, cwa_t, swa_t, cpx_t, preds_lbl, true_lbl = run_eval(
    test_loader, spr_raw["test"]["sequence"]
)
print(f"TEST  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}")

ed = experiment_data["No-Pos"]["SPR_BENCH"]
ed["metrics"]["test"] = {"cwa": cwa_t, "swa": swa_t, "cpxwa": cpx_t}
ed["predictions"] = preds_lbl
ed["ground_truth"] = true_lbl

# ------------------- SAVE ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
