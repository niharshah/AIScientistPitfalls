import os, pathlib, random, itertools
import numpy as np, torch, torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from datasets import load_dataset, DatasetDict
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# -------------------- experiment bookkeeping --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
ablation_key = "no_shape"  # <── current ablation name
dataset_key = "SPR_BENCH"
experiment_data = {
    ablation_key: {
        dataset_key: {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- helpers --------------------
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
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
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


# -------------------- graph creation --------------------
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
        for i in range(n - 1):
            for s, d in ((i, i + 1), (i + 1, i)):
                edge_src.append(s)
                edge_dst.append(d)
                edge_type.append(0)
        groups = defaultdict(list)
        for idx, tok in enumerate(toks):
            groups[("shape", tok[0])].append(idx)
            groups[("color", tok[1])].append(idx)
        for gkey, idxs in groups.items():
            rel = 1 if gkey[0] == "shape" else 2
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_type.append(rel)
        if not edge_src:
            edge_src, edge_dst, edge_type = [0], [0], [0]
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        x = torch.stack([shape_idx, color_idx, pos_idx], dim=1)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([lab2i[label]]),
        )

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = [
            seq_to_graph(seq, lab)
            for seq, lab in zip(spr_dict[split]["sequence"], spr_dict[split]["label"])
        ]
    return out, shape2i, color2i, lab2i


# -------------------- datasets --------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr_raw = load_spr_bench(DATA_PATH)
else:

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

train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
val_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)


# -------------------- model (NO-SHAPE EMBEDDING) --------------------
class RGCNClassifier_NoShape(nn.Module):
    def __init__(self, n_color, max_pos, emb_dim, hid_dim, n_class):
        super().__init__()
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(max_pos, emb_dim)
        self.conv1 = RGCNConv(emb_dim, hid_dim, num_relations=3)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_relations=3)
        self.lin = nn.Linear(hid_dim, n_class)

    def forward(self, data):
        _, c, p = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        h = self.color_emb(c) + self.pos_emb(p)  # ← no shape term
        h = self.conv1(h, data.edge_index, data.edge_type).relu()
        h = self.conv2(h, data.edge_index, data.edge_type).relu()
        h = global_mean_pool(h, data.batch)
        return self.lin(h)


model = RGCNClassifier_NoShape(
    len(color2i), max_pos, emb_dim=32, hid_dim=64, n_class=len(lab2i)
).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
EPOCHS = 25
best_val_cpx, best_state = -1, None


# -------------------- evaluation --------------------
def run_eval(loader, seqs):
    model.eval()
    tot_loss, pr, gt = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            tot_loss += loss.item() * batch.num_graphs
            pr.extend(out.argmax(-1).cpu().tolist())
            gt.extend(batch.y.cpu().tolist())
    avg = tot_loss / len(loader.dataset)
    pr_lbl = [inv_lab[p] for p in pr]
    gt_lbl = [inv_lab[t] for t in gt]
    cwa = color_weighted_accuracy(seqs, gt_lbl, pr_lbl)
    swa = shape_weighted_accuracy(seqs, gt_lbl, pr_lbl)
    cpx = complexity_weighted_accuracy(seqs, gt_lbl, pr_lbl)
    return avg, cwa, swa, cpx, pr_lbl, gt_lbl


# -------------------- train --------------------
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

    experiment_data[ablation_key][dataset_key]["losses"]["train"].append(train_loss)
    experiment_data[ablation_key][dataset_key]["losses"]["val"].append(val_loss)
    experiment_data[ablation_key][dataset_key]["metrics"]["val"].append(
        {"epoch": epoch, "cwa": cwa, "swa": swa, "cpxwa": cpx}
    )
    if cpx > best_val_cpx:
        best_val_cpx, cx = cpx, None
        best_state = model.state_dict()

# -------------------- test --------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, cwa_t, swa_t, cpx_t, preds_lbl, true_lbl = run_eval(
    test_loader, spr_raw["test"]["sequence"]
)
print(f"TEST  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}")

experiment_data[ablation_key][dataset_key]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cpxwa": cpx_t,
}
experiment_data[ablation_key][dataset_key]["predictions"] = preds_lbl
experiment_data[ablation_key][dataset_key]["ground_truth"] = true_lbl

# -------------------- save --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
