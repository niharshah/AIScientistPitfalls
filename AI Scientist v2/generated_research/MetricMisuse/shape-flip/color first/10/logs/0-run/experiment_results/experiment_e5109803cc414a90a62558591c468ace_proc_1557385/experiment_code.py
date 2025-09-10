import os, pathlib, random, itertools, time, numpy as np, torch, torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from datasets import load_dataset, DatasetDict, Dataset
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ------------------- EXPERIMENT LOG -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "ConcatEmb_NoEarlyFusion": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------- DEVICE ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------- DATA HELPERS ---------------------
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
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / (sum(w) or 1)


# ------------------- GRAPH CONSTRUCTION ---------------
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
        s_idx = torch.tensor([shape2i[t[0]] for t in toks])
        c_idx = torch.tensor([color2i[t[1]] for t in toks])
        p_idx = torch.tensor(range(n))
        edge_src, edge_dst, edge_type = [], [], []
        # sequential neighbours (rel 0)
        for i in range(n - 1):
            for a, b in ((i, i + 1), (i + 1, i)):
                edge_src.append(a)
                edge_dst.append(b)
                edge_type.append(0)
        # same-shape (rel 1) & same-col (rel 2)
        groups = defaultdict(list)
        for i, t in enumerate(toks):
            groups[("shape", t[0])].append(i)
            groups[("color", t[1])].append(i)
        for (kind, _), idxs in groups.items():
            rel = 1 if kind == "shape" else 2
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_type.append(rel)
        if not edge_src:
            edge_src, edge_dst, edge_type = [0], [0], [0]
        data = Data(
            x=torch.stack([s_idx, c_idx, p_idx], 1),
            edge_index=torch.tensor([edge_src, edge_dst]),
            edge_type=torch.tensor(edge_type),
            y=torch.tensor([lab2i[label]]),
        )
        return data

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = [
            seq_to_graph(s, l)
            for s, l in zip(spr_dict[split]["sequence"], spr_dict[split]["label"])
        ]
    return out, shape2i, color2i, lab2i


# ------------------- LOAD / SYNTHETIC DATA ------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr_raw = load_spr_bench(DATA_PATH)
else:  # tiny synthetic fallback

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
num_class = len(lab2i)

train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
val_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)


# ------------------- MODEL ----------------------------
class RGCNClassifierConcat(nn.Module):
    def __init__(self, n_shape, n_color, max_pos, emb_dim, hid_dim, n_class):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(max_pos, emb_dim)
        self.conv1 = RGCNConv(emb_dim * 3, hid_dim, num_relations=3)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_relations=3)
        self.lin = nn.Linear(hid_dim, n_class)

    def forward(self, data):
        s, c, p = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        h = torch.cat([self.shape_emb(s), self.color_emb(c), self.pos_emb(p)], dim=-1)
        h = self.conv1(h, data.edge_index, data.edge_type).relu()
        h = self.conv2(h, data.edge_index, data.edge_type).relu()
        h = global_mean_pool(h, data.batch)
        return self.lin(h)


# ------------------- TRAINING SETUP -------------------
model = RGCNClassifierConcat(
    len(shape2i), len(color2i), max_pos, emb_dim=32, hid_dim=64, n_class=num_class
).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
EPOCHS = 25
best_val_cpx, best_state = -1, None


# ------------------- EVAL FUNCTION --------------------
def run_eval(loader, seqs):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
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


# ------------------- TRAIN LOOP -----------------------
for ep in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.num_graphs
    train_loss = running_loss / len(train_loader.dataset)

    v_loss, cwa, swa, cpx, _, _ = run_eval(val_loader, spr_raw["dev"]["sequence"])
    print(
        f"Epoch {ep} | val_loss {v_loss:.4f} | CWA {cwa:.3f} SWA {swa:.3f} CpxWA {cpx:.3f}"
    )

    experiment_data["ConcatEmb_NoEarlyFusion"]["SPR_BENCH"]["losses"]["train"].append(
        train_loss
    )
    experiment_data["ConcatEmb_NoEarlyFusion"]["SPR_BENCH"]["losses"]["val"].append(
        v_loss
    )
    experiment_data["ConcatEmb_NoEarlyFusion"]["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": ep, "cwa": cwa, "swa": swa, "cpxwa": cpx}
    )
    if cpx > best_val_cpx:
        best_val_cpx = cpx
        best_state = model.state_dict()

# ------------------- TEST EVAL ------------------------
if best_state is not None:
    model.load_state_dict(best_state)
t_loss, cwa_t, swa_t, cpx_t, preds_lbl, true_lbl = run_eval(
    test_loader, spr_raw["test"]["sequence"]
)
print(f"TEST  CWA {cwa_t:.3f} | SWA {swa_t:.3f} | CpxWA {cpx_t:.3f}")

exp_entry = experiment_data["ConcatEmb_NoEarlyFusion"]["SPR_BENCH"]
exp_entry["metrics"]["test"] = {"cwa": cwa_t, "swa": swa_t, "cpxwa": cpx_t}
exp_entry["predictions"] = preds_lbl
exp_entry["ground_truth"] = true_lbl

# ------------------- SAVE -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
