import os, random, string, math, time, json, warnings, pathlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_color_variety(seq):  # token[1] = colour
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):  # token[0] = shape
    return len(set(t[0] for t in seq.split() if t))


def color_weighted_accuracy(seqs, y, yhat):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y, yhat)]
    return sum(c) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y, yhat):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y, yhat)]
    return sum(c) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ---------- dataset ----------
def load_spr_bench_if_present():
    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None
    from datasets import load_dataset, DatasetDict

    def ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = {"train": ld("train.csv"), "dev": ld("dev.csv"), "test": ld("test.csv")}
    from datasets import DatasetDict

    return DatasetDict(d)


def make_synthetic(n_tr=400, n_dev=120, n_te=150):
    shapes = list(string.ascii_uppercase[:8])
    colors = list(string.ascii_lowercase[:8])

    def gen(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 15)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seqs.append(" ".join(toks))
            # simple majority-shape rule
            labels.append(int(sum(t[0] == toks[0][0] for t in toks) > L / 2))
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(gen(n_tr)),
            "dev": Dataset.from_dict(gen(n_dev)),
            "test": Dataset.from_dict(gen(n_te)),
        }
    )


dataset = load_spr_bench_if_present() or make_synthetic()
print("Dataset sizes:", {k: len(v) for k, v in dataset.items()})

# ---------- vocab ----------
shape2idx, color2idx = {}, {}


def add_tok(tok):
    if tok[0] not in shape2idx:
        shape2idx[tok[0]] = len(shape2idx)
    if tok[1] not in color2idx:
        color2idx[tok[1]] = len(color2idx)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_tok(tok)
n_shapes, n_colors = len(shape2idx), len(color2idx)
num_classes = len(set(dataset["train"]["label"]))
print(f"Vocab: shapes={n_shapes} colours={n_colors}  classes={num_classes}")


# ---------- graph construction ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shape_ids = [shape2idx[t[0]] for t in toks]
    color_ids = [color2idx[t[1]] for t in toks]

    # relation 0: sequential
    src0 = list(range(n - 1))
    dst0 = list(range(1, n))
    # relation 1: same-shape
    src1, dst1 = [], []
    bucket = {}
    for i, s in enumerate(shape_ids):
        bucket.setdefault(s, []).append(i)
    for ids in bucket.values():
        for i in ids:
            for j in ids:
                if i < j:
                    src1.append(i)
                    dst1.append(j)
    # relation 2: same-colour
    src2, dst2 = [], []
    bucket = {}
    for i, c in enumerate(color_ids):
        bucket.setdefault(c, []).append(i)
    for ids in bucket.values():
        for i in ids:
            for j in ids:
                if i < j:
                    src2.append(i)
                    dst2.append(j)

    def mk(src, dst):
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
        return ei

    data = Data(
        x=torch.tensor(list(zip(shape_ids, color_ids)), dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
        edge_index_seq=mk(src0, dst0),
        edge_index_shape=mk(src1, dst1),
        edge_index_color=mk(src2, dst2),
    )
    data.seq_raw = seq
    return data


graphs = {
    split: [
        seq_to_graph(s, l)
        for s, l in zip(dataset[split]["sequence"], dataset[split]["label"])
    ]
    for split in ["train", "dev", "test"]
}


# ---------- model ----------
class MultiRelSAGE(nn.Module):
    def __init__(self, in_dim, out_dim, num_rel=3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, out_dim) for _ in range(num_rel)])

    def forward(self, x, edge_indices):
        out = 0
        for conv, ei in zip(self.convs, edge_indices):
            if ei.numel():
                out = out + conv(x, ei)
        return out


class GNN(nn.Module):
    def __init__(self, n_shapes, n_colors, emb_dim=32, hid=64, n_cls=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb_dim)
        self.color_emb = nn.Embedding(n_colors, emb_dim)
        self.lin0 = nn.Linear(emb_dim * 2, hid)
        self.rel1 = MultiRelSAGE(hid, hid)
        self.rel2 = MultiRelSAGE(hid, hid)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hid, n_cls)

    def forward(self, batch):
        sh = self.shape_emb(batch.x[:, 0])
        co = self.color_emb(batch.x[:, 1])
        x = F.relu(self.lin0(torch.cat([sh, co], dim=-1)))
        ei = [batch.edge_index_seq, batch.edge_index_shape, batch.edge_index_color]
        x = F.relu(self.rel1(x, ei))
        x = self.dropout(F.relu(self.rel2(x, ei)))
        x = global_mean_pool(x, batch.batch)
        return self.out(x)


# ---------- logging dict ----------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- training / evaluation ----------
def evaluate(model, loader, crit):
    model.eval()
    tot_loss, preds, gts, seqs = 0, [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits = model(b)
            loss = crit(logits, b.y)
            tot_loss += loss.item() * b.num_graphs
            p = logits.argmax(-1).cpu().tolist()
            g = b.y.cpu().tolist()
            preds.extend(p)
            gts.extend(g)
            seqs.extend(b.seq_raw)
    tot_loss /= len(loader.dataset)
    acc = np.mean([p == g for p, g in zip(preds, gts)])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hpa = harmonic_poly_accuracy(cwa, swa)
    return tot_loss, acc, cwa, swa, hpa, preds, gts, seqs


def train_one(lr, emb=32, hid=64, epochs=10):
    print(f"\n=== LR={lr} emb={emb} hid={hid} ===")
    model = GNN(n_shapes, n_colors, emb_dim=emb, hid=hid, n_cls=num_classes).to(device)
    optim = Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(graphs["train"], batch_size=32, shuffle=True)
    dl = DataLoader(graphs["dev"], batch_size=64)
    best_hpa, best_state = -1, None
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0
        for b in tl:
            b = b.to(device)
            optim.zero_grad()
            loss = crit(model(b), b.y)
            loss.backward()
            optim.step()
            tot += loss.item() * b.num_graphs
        tr_loss = tot / len(tl.dataset)
        val_loss, acc, cwa, swa, hpa, *_ = evaluate(model, dl, crit)
        print(
            f"Epoch {ep}: validation_loss = {val_loss:.4f} | acc={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}"
        )
        # log
        experiment_data["SPR"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR"]["losses"]["val"].append(val_loss)
        experiment_data["SPR"]["metrics"]["val"].append(
            {"acc": acc, "CWA": cwa, "SWA": swa, "HPA": hpa}
        )
        experiment_data["SPR"]["epochs"].append(ep)
        if hpa > best_hpa:
            best_hpa, best_state = hpa, model.state_dict()
    # restore best
    model.load_state_dict(best_state)
    return model


best_model = None
best_hpa = -1
for lr in [1e-3, 3e-3]:
    mdl = train_one(lr)
    hpa = experiment_data["SPR"]["metrics"]["val"][-1]["HPA"]
    if hpa > best_hpa:
        best_hpa, best_model = hpa, mdl

# ---------- final test evaluation ----------
test_loader = DataLoader(graphs["test"], batch_size=64)
crit = nn.CrossEntropyLoss()
test_loss, acc, cwa, swa, hpa, preds, gts, seqs = evaluate(
    best_model, test_loader, crit
)
print(
    f"\nTEST  loss={test_loss:.4f} acc={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}"
)

experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data ->", os.path.join(working_dir, "experiment_data.npy"))
