# Single-Relation GCN (no relation types)
import os, pathlib, random, itertools, time, numpy as np, torch, torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from datasets import load_dataset, DatasetDict
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- save dict ----------
experiment_data = {
    "SingleRelGCN": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- misc ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------- helpers ----------
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


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / (sum(w) or 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / (sum(w) or 1)


# ---------- graph dataset ----------
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
        x = torch.stack(
            [
                torch.tensor([shape2i[t[0]] for t in toks]),
                torch.tensor([color2i[t[1]] for t in toks]),
                torch.tensor(list(range(n))),
            ],
            dim=1,
        )
        e_src, e_dst = [], []
        # sequential
        for i in range(n - 1):
            for s, d in ((i, i + 1), (i + 1, i)):
                e_src.append(s)
                e_dst.append(d)
        # same shape / same color
        groups = defaultdict(list)
        for idx, tok in enumerate(toks):
            groups[("shape", tok[0])].append(idx)
            groups[("color", tok[1])].append(idx)
        for idxs in groups.values():
            for i in idxs:
                for j in idxs:
                    if i != j:
                        e_src.append(i)
                        e_dst.append(j)
        if not e_src:
            e_src, e_dst = [0], [0]  # 1-node safeguard
        edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
        return Data(
            x=x, edge_index=edge_index, y=torch.tensor([lab2i[label]], dtype=torch.long)
        )

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = [
            seq_to_graph(seq, lab)
            for seq, lab in zip(spr_dict[split]["sequence"], spr_dict[split]["label"])
        ]
    return out, shape2i, color2i, lab2i


# ---------- load data ----------
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
num_class = len(lab2i)

train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
val_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)


# ---------- model ----------
class GCNClassifier(nn.Module):
    def __init__(self, n_shape, n_color, max_pos, emb_dim, hid_dim, n_class):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(max_pos, emb_dim)
        self.conv1 = GCNConv(emb_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.lin = nn.Linear(hid_dim, n_class)

    def forward(self, data):
        s, c, p = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        h = self.shape_emb(s) + self.color_emb(c) + self.pos_emb(p)
        h = self.conv1(h, data.edge_index).relu()
        h = self.conv2(h, data.edge_index).relu()
        h = global_mean_pool(h, data.batch)
        return self.lin(h)


model = GCNClassifier(
    len(shape2i), len(color2i), max_pos, emb_dim=32, hid_dim=64, n_class=num_class
).to(device)
opt = Adam(model.parameters(), lr=1e-3)


# ---------- eval ----------
def run_eval(loader, seqs):
    model.eval()
    loss_sum = 0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = cross_entropy(out, batch.y.view(-1))
            loss_sum += loss.item() * batch.num_graphs
            preds.extend(out.argmax(-1).cpu().tolist())
            trues.extend(batch.y.cpu().tolist())
    avg_loss = loss_sum / len(loader.dataset)
    pred_lbl = [inv_lab[p] for p in preds]
    true_lbl = [inv_lab[t] for t in trues]
    return (
        avg_loss,
        color_weighted_accuracy(seqs, true_lbl, pred_lbl),
        shape_weighted_accuracy(seqs, true_lbl, pred_lbl),
        complexity_weighted_accuracy(seqs, true_lbl, pred_lbl),
        pred_lbl,
        true_lbl,
    )


# ---------- train ----------
EPOCHS = 25
best_cpx = -1
best_state = None
for ep in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch)
        loss = cross_entropy(out, batch.y.view(-1))
        loss.backward()
        opt.step()
        ep_loss += loss.item() * batch.num_graphs
    tr_loss = ep_loss / len(train_loader.dataset)

    v_loss, cwa, swa, cpx, _, _ = run_eval(val_loader, spr_raw["dev"]["sequence"])
    print(
        f"Epoch {ep}  ValLoss {v_loss:.4f}  CWA {cwa:.3f}  SWA {swa:.3f}  CpxWA {cpx:.3f}"
    )

    experiment_data["SingleRelGCN"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SingleRelGCN"]["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["SingleRelGCN"]["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": ep, "cwa": cwa, "swa": swa, "cpxwa": cpx}
    )

    if cpx > best_cpx:
        best_cpx = cpx
        best_state = model.state_dict()

# ---------- test ----------
if best_state is not None:
    model.load_state_dict(best_state)
t_loss, cwa_t, swa_t, cpx_t, pred_lbl, true_lbl = run_eval(
    test_loader, spr_raw["test"]["sequence"]
)
print(f"TEST  CWA {cwa_t:.3f}  SWA {swa_t:.3f}  CpxWA {cpx_t:.3f}")

experiment_data["SingleRelGCN"]["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cpxwa": cpx_t,
}
experiment_data["SingleRelGCN"]["SPR_BENCH"]["predictions"] = pred_lbl
experiment_data["SingleRelGCN"]["SPR_BENCH"]["ground_truth"] = true_lbl

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
