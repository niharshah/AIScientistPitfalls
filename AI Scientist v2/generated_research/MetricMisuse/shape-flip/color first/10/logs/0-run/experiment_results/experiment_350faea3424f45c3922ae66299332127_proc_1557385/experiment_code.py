import os, pathlib, random, itertools, time
import numpy as np
import torch, torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from datasets import load_dataset, DatasetDict
from collections import defaultdict

# ---------------- experiment log dict -----------------
experiment_data = {
    "sequential_only": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- DATA HELPERS ----------------
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


def count_color_variety(seq):  # CWA weight
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):  # SWA weight
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


# ---------------- GRAPH CONSTRUCTION (SEQUENTIAL ONLY) ----------------
from torch_geometric.data import Data


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
    shape2i = {s: i for i, s in enumerate(shapes)}
    color2i = {c: i for i, c in enumerate(colors)}
    lab2i = {l: i for i, l in enumerate(labels)}

    def seq_to_graph(seq, label):
        toks = seq.split()
        n = len(toks)
        shape_idx = torch.tensor([shape2i[t[0]] for t in toks], dtype=torch.long)
        color_idx = torch.tensor([color2i[t[1]] for t in toks], dtype=torch.long)
        pos_idx = torch.tensor(list(range(n)), dtype=torch.long)
        edge_src, edge_dst, edge_type = [], [], []
        if n == 1:
            edge_src = [0]
            edge_dst = [0]
            edge_type = [0]
        else:
            for i in range(n - 1):
                edge_src.extend([i, i + 1])
                edge_dst.extend([i + 1, i])
                edge_type.extend([0, 0])  # only relation 0
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        x = torch.stack([shape_idx, color_idx, pos_idx], dim=1)
        return Data(
            x=x,
            edge_index=edge_index,
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


# ---------------- MODEL ----------------
from torch_geometric.nn import RGCNConv, global_mean_pool


class RGCNClassifier(nn.Module):
    def __init__(self, n_shape, n_color, max_pos, emb_dim, hid_dim, n_class):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(max_pos, emb_dim)
        self.conv1 = RGCNConv(emb_dim, hid_dim, num_relations=1)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_relations=1)
        self.lin = nn.Linear(hid_dim, n_class)

    def forward(self, data):
        s, c, p = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        h = self.shape_emb(s) + self.color_emb(c) + self.pos_emb(p)
        h = self.conv1(h, data.edge_index, data.edge_type).relu()
        h = self.conv2(h, data.edge_index, data.edge_type).relu()
        h = global_mean_pool(h, data.batch)
        return self.lin(h)


# ---------------- LOAD DATA ----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr_raw = load_spr_bench(DATA_PATH)
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

from torch_geometric.loader import DataLoader

train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
val_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)

# ---------------- INIT MODEL ----------------
model = RGCNClassifier(
    len(shape2i), len(color2i), max_pos, emb_dim=32, hid_dim=64, n_class=len(lab2i)
).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
EPOCHS = 25
best_val_cpx, best_state = -1, None


# ---------------- EVAL FUNC ----------------
def run_eval(loader, seqs):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.argmax(-1).cpu().tolist())
            trues.extend(batch.y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    pred_lbl = [inv_lab[p] for p in preds]
    true_lbl = [inv_lab[t] for t in trues]
    cwa = color_weighted_accuracy(seqs, true_lbl, pred_lbl)
    swa = shape_weighted_accuracy(seqs, true_lbl, pred_lbl)
    cpx = complexity_weighted_accuracy(seqs, true_lbl, pred_lbl)
    return avg_loss, cwa, swa, cpx, pred_lbl, true_lbl


# ---------------- TRAIN LOOP ----------------
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
    # logging
    d = experiment_data["sequential_only"]["SPR_BENCH"]
    d["losses"]["train"].append(train_loss)
    d["losses"]["val"].append(val_loss)
    d["metrics"]["val"].append({"epoch": epoch, "cwa": cwa, "swa": swa, "cpxwa": cpx})
    if cpx > best_val_cpx:
        best_val_cpx = cpx
        best_state = model.state_dict()

# ---------------- TEST EVAL ----------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, cwa_t, swa_t, cpx_t, preds_lbl, tru_lbl = run_eval(
    test_loader, spr_raw["test"]["sequence"]
)
print(f"TEST  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}")
d = experiment_data["sequential_only"]["SPR_BENCH"]
d["metrics"]["test"] = {"cwa": cwa_t, "swa": swa_t, "cpxwa": cpx_t}
d["predictions"] = preds_lbl
d["ground_truth"] = tru_lbl

# ---------------- SAVE RESULTS ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
