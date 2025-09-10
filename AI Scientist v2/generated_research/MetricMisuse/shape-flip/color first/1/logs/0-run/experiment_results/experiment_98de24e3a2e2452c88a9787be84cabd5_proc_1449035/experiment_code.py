import os, random, string, warnings, math, pathlib, time, numpy as np, torch
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
    return len(set(tok[1] for tok in seq.split()))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ---------------- dataset generation ------------------
shapes = list(string.ascii_uppercase[:8])  # A–H
colors = list(string.ascii_lowercase[:8])  # a–h


def gen_sequence(L=None):
    L = L or random.randint(4, 15)
    return [random.choice(shapes) + random.choice(colors) for _ in range(L)]


def label_rule(seq, rule):
    if rule == "maj_shape":
        first = seq[0][0]
        return int(sum(tok[0] == first for tok in seq) > len(seq) / 2)
    if rule == "maj_color":
        first = seq[0][1]
        return int(sum(tok[1] == first for tok in seq) > len(seq) / 2)
    # parity_of_first: even->1, odd->0 (shape idx + color idx)
    si, ci = shapes.index(seq[0][0]), colors.index(seq[0][1])
    return int((si + ci) % 2 == 0)


def make_dataset(rule, n_tr=600, n_dev=200, n_te=250):
    def _make(n):
        seqs, labels = [], []
        for _ in range(n):
            toks = gen_sequence()
            seqs.append(" ".join(toks))
            labels.append(label_rule(toks, rule))
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(_make(n_tr)),
            "dev": Dataset.from_dict(_make(n_dev)),
            "test": Dataset.from_dict(_make(n_te)),
        }
    )


rules = ["maj_shape", "maj_color", "parity_first"]
datasets = {r: make_dataset(r) for r in rules}
print({k: {split: len(v[split]) for split in v} for k, v in datasets.items()})

# ---------------- global vocab ------------------------
shape2idx, color2idx = {}, {}
for r in rules:
    for seq in datasets[r]["train"]["sequence"]:
        for tok in seq.split():
            s, c = tok[0], tok[1]
            if s not in shape2idx:
                shape2idx[s] = len(shape2idx)
            if c not in color2idx:
                color2idx[c] = len(color2idx)
n_shapes, n_colors = len(shape2idx), len(color2idx)
print("Vocab:", n_shapes, n_colors)


# ---------------- seq -> graph ------------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    s_ids = [shape2idx[t[0]] for t in toks]
    c_ids = [color2idx[t[1]] for t in toks]
    x = torch.tensor(list(zip(s_ids, c_ids)), dtype=torch.long)
    src, dst, etype = [], [], []
    # consecutive
    for i in range(len(toks) - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [0, 0]
    # same-shape
    bucket = {}
    for i, s in enumerate(s_ids):
        bucket.setdefault(s, []).append(i)
    for nodes in bucket.values():
        for i in nodes:
            for j in nodes:
                if i < j:
                    src += [i, j]
                    dst += [j, i]
                    etype += [1, 1]
    # same-color
    bucket = {}
    for i, c in enumerate(c_ids):
        bucket.setdefault(c, []).append(i)
    for nodes in bucket.values():
        for i in nodes:
            for j in nodes:
                if i < j:
                    src += [i, j]
                    dst += [j, i]
                    etype += [2, 2]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([label]))
    d.seq_raw = seq
    return d


# ---------------- model --------------------------------
class RGCNClassifier(nn.Module):
    def __init__(self, n_shapes, n_colors, n_rel=3, emb=32, hid=64, n_cls=2):
        super().__init__()
        self.se = nn.Embedding(n_shapes, emb)
        self.ce = nn.Embedding(n_colors, emb)
        self.lin = nn.Linear(emb * 2, hid)
        self.c1 = RGCNConv(hid, hid, n_rel)
        self.c2 = RGCNConv(hid, hid, n_rel)
        self.out = nn.Linear(hid, n_cls)

    def forward(self, data):
        x = torch.cat([self.se(data.x[:, 0]), self.ce(data.x[:, 1])], dim=-1)
        x = F.relu(self.lin(x))
        x = F.relu(self.c1(x, data.edge_index, data.edge_type))
        x = F.relu(self.c2(x, data.edge_index, data.edge_type))
        return self.out(global_mean_pool(x, data.batch))


# ---------------- experiment logger --------------------
experiment_data = {
    "MultiSynthetic": {
        r: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "cross_test": {},
            "predictions": {},
            "ground_truth": {},
        }
        for r in rules
    }
}


# ---------------- helper -------------------------------
def build_loader(graphs, bs, shuffle=False):
    return DataLoader(graphs, batch_size=bs, shuffle=shuffle)


# ---------------- training & evaluation ----------------
EPOCHS = 12
for src_rule in rules:
    print(f"\n=== Train on {src_rule} ===")
    # graphs
    tr_graphs = [
        seq_to_graph(s, l)
        for s, l in zip(
            datasets[src_rule]["train"]["sequence"],
            datasets[src_rule]["train"]["label"],
        )
    ]
    dv_graphs = [
        seq_to_graph(s, l)
        for s, l in zip(
            datasets[src_rule]["dev"]["sequence"], datasets[src_rule]["dev"]["label"]
        )
    ]
    loaders = {
        "train": build_loader(tr_graphs, 32, True),
        "dev": build_loader(dv_graphs, 64),
    }
    num_cls = len(set(datasets[src_rule]["train"]["label"]))
    # class weights
    cw = torch.tensor(datasets[src_rule]["train"]["label"])
    w = 1.0 / (torch.bincount(cw).float() + 1e-6)
    w = w / w.sum() * num_cls
    w = w.to(device)
    model = RGCNClassifier(n_shapes, n_colors, n_cls=num_cls).to(device)
    opt = Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS + 1)
    crit = nn.CrossEntropyLoss(weight=w)
    best_hpa, best_state = -1, None
    # loop
    for ep in range(1, EPOCHS + 1):
        model.train()
        tloss = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            loss = crit(model(batch), batch.y)
            loss.backward()
            opt.step()
            tloss += loss.item() * batch.num_graphs
        tloss /= len(loaders["train"].dataset)
        experiment_data["MultiSynthetic"][src_rule]["losses"]["train"].append(tloss)
        # dev
        model.eval()
        vloss = 0
        preds = gts = seqs = []
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in loaders["dev"]:
                batch = batch.to(device)
                logits = model(batch)
                vloss += crit(logits, batch.y).item() * batch.num_graphs
                preds += logits.argmax(-1).cpu().tolist()
                gts += batch.y.cpu().tolist()
                seqs += batch.seq_raw
        vloss /= len(loaders["dev"].dataset)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hpa = harmonic_poly_accuracy(cwa, swa)
        experiment_data["MultiSynthetic"][src_rule]["losses"]["val"].append(vloss)
        experiment_data["MultiSynthetic"][src_rule]["metrics"]["val"].append(
            {"CWA": cwa, "SWA": swa, "HPA": hpa}
        )
        if hpa > best_hpa:
            best_hpa, best_state = hpa, {
                k: v.cpu() for k, v in model.state_dict().items()
            }
        sched.step()
        print(f"Ep{ep:02d} TL={tloss:.3f} VL={vloss:.3f} HPA={hpa:.3f}")
    # reload best
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    # cross-dataset test
    for tgt_rule in rules:
        tg_graphs = [
            seq_to_graph(s, l)
            for s, l in zip(
                datasets[tgt_rule]["test"]["sequence"],
                datasets[tgt_rule]["test"]["label"],
            )
        ]
        loader = build_loader(tg_graphs, 64)
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                preds += logits.argmax(-1).cpu().tolist()
                gts += batch.y.cpu().tolist()
                seqs += batch.seq_raw
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hpa = harmonic_poly_accuracy(cwa, swa)
        experiment_data["MultiSynthetic"][src_rule]["cross_test"][tgt_rule] = {
            "CWA": cwa,
            "SWA": swa,
            "HPA": hpa,
        }
        experiment_data["MultiSynthetic"][src_rule]["predictions"][tgt_rule] = preds
        experiment_data["MultiSynthetic"][src_rule]["ground_truth"][tgt_rule] = gts
        print(f"  Test on {tgt_rule:12s} | CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}")

# ---------------- save -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved ->", os.path.join(working_dir, "experiment_data.npy"))
