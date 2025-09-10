import os, random, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---------- misc ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------- helpers ----------
def uniq_shapes(seq):
    return len({tok[0] for tok in seq.split()})


def uniq_colors(seq):
    return len({tok[1] for tok in seq.split()})


# three-class synthetic labelling rules
def label_variety(seq):
    us, uc = uniq_shapes(seq), uniq_colors(seq)
    return 0 if us > uc else 1 if us == uc else 2


def label_freq(seq):
    shapes = [tok[0] for tok in seq.split()]
    colors = [tok[1] for tok in seq.split()]
    fs = max(shapes.count(s) for s in set(shapes))
    fc = max(colors.count(c) for c in set(colors))
    return 0 if fs > fc else 1 if fs == fc else 2


def label_mod(seq):
    return len(seq.split()) % 3  # 0 / 1 / 2


rule2labfn = {"variety": label_variety, "freq": label_freq, "mod": label_mod}


def synth_dataset(rule: str, n: int) -> Dict[str, List]:
    shapes, colors = list("ABCD"), list("1234")
    seqs, labs = [], []
    for _ in range(n):
        ln = np.random.randint(4, 9)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(toks)
        seqs.append(seq)
        labs.append(rule2labfn[rule](seq))
    return {"id": list(range(n)), "sequence": seqs, "label": labs}


# ---------- make all datasets ----------
splits = {"train": 8000, "dev": 2000, "test": 2000}
rules = ["variety", "freq", "mod"]
raw_data = {r: {sp: synth_dataset(r, n) for sp, n in splits.items()} for r in rules}

# union (mixed rules) dataset
raw_data["union"] = {}
for sp in splits:
    seq, lab = [], []
    for r in rules:
        seq += raw_data[r][sp]["sequence"]
        lab += raw_data[r][sp]["label"]
    raw_data["union"][sp] = {"id": list(range(len(seq))), "sequence": seq, "label": lab}

# ---------- vocab ----------
all_shapes = set("ABCD")
all_colors = set("1234")
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
num_shapes, num_colors, num_class = len(shape2idx), len(color2idx), 3


# ---------- graph builder ----------
def seq_to_graph(seq: str, lab: int) -> Data:
    toks = seq.split()
    sh = torch.tensor([shape2idx[t[0]] for t in toks], dtype=torch.long)
    co = torch.tensor([color2idx[t[1]] for t in toks], dtype=torch.long)
    x = torch.stack([sh, co], 1)
    src, dst, etype = [], [], []
    # order edges
    for i in range(len(toks) - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [0, 0]
    # relational edges
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            if co[i] == co[j]:
                src += [i, j]
                dst += [j, i]
                etype += [1, 1]
            if sh[i] == sh[j]:
                src += [i, j]
                dst += [j, i]
                etype += [2, 2]
    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.tensor(etype, dtype=torch.long),
        y=torch.tensor([lab], dtype=torch.long),
        seq=seq,
    )
    return data


def build_pyg_dataset(blob):
    return [seq_to_graph(s, l) for s, l in zip(blob["sequence"], blob["label"])]


pyg_data = {
    r: {sp: build_pyg_dataset(raw_data[r][sp]) for sp in splits} for r in raw_data
}


# ---------- model ----------
class RModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.se = nn.Embedding(num_shapes, 8)
        self.ce = nn.Embedding(num_colors, 8)
        self.pre = nn.Linear(16, 32)
        self.c1 = RGCNConv(32, 64, num_relations=3)
        self.c2 = RGCNConv(64, 64, num_relations=3)
        self.cls = nn.Linear(64, num_class)

    def forward(self, batch):
        x = torch.cat([self.se(batch.x[:, 0]), self.ce(batch.x[:, 1])], 1)
        x = F.relu(self.pre(x))
        x = F.relu(self.c1(x, batch.edge_index, batch.edge_type))
        x = F.relu(self.c2(x, batch.edge_index, batch.edge_type))
        x = global_mean_pool(x, batch.batch)
        return self.cls(x)


# ---------- training / evaluation ----------
def run_training(ds_name, epochs=10):
    mdl = RModel().to(device)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tl = DataLoader(pyg_data[ds_name]["train"], batch_size=64, shuffle=True)
    vl = DataLoader(pyg_data[ds_name]["dev"], batch_size=128, shuffle=False)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        mdl.train()
        tot = 0
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            out = mdl(b)
            loss = loss_fn(out, b.y)
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
        tr_loss = tot / len(tl.dataset)
        mdl.eval()
        tot = 0
        correct = 0
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                out = mdl(b)
                tot += loss_fn(out, b.y).item() * b.num_graphs
                correct += (out.argmax(1) == b.y).sum().item()
        vl_loss = tot / len(vl.dataset)
        acc = correct / len(vl.dataset)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["val_acc"].append(acc)
        print(f"[{ds_name}] epoch {ep}: val_loss={vl_loss:.4f} acc={acc:.3f}")
    return mdl, hist


def evaluate(mdl, ds_name):
    mdl.eval()
    dl = DataLoader(pyg_data[ds_name]["test"], batch_size=128, shuffle=False)
    preds, gts = [], []
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            out = mdl(b)
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(b.y.cpu().tolist())
    acc = sum(int(p == t) for p, t in zip(preds, gts)) / len(gts)
    return acc, preds, gts


# ---------- run all trainings ----------
experiment_data = {"multi_rule_ablation": defaultdict(dict)}
models = {}
for ds in ["variety", "freq", "mod", "union"]:
    mdl, hist = run_training(ds)
    models[ds] = mdl
    experiment_data["multi_rule_ablation"][ds]["losses"] = hist
    # predictions will be filled later

# ---------- transfer evaluation ----------
for tr_name, mdl in models.items():
    res = {}
    for te_name in ["variety", "freq", "mod"]:
        acc, preds, gts = evaluate(mdl, te_name)
        res[te_name] = {"accuracy": acc}
        if tr_name == te_name:  # store self-predictions for further plotting
            experiment_data["multi_rule_ablation"][te_name]["predictions"] = preds
            experiment_data["multi_rule_ablation"][te_name]["ground_truth"] = gts
        print(f"Model[{tr_name}] -> Test[{te_name}]  acc={acc:.3f}")
    experiment_data["multi_rule_ablation"][tr_name]["transfer_acc"] = res

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ---------- quick plot ----------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure()
for ds in ["variety", "freq", "mod", "union"]:
    plt.plot(experiment_data["multi_rule_ablation"][ds]["losses"]["val_acc"], label=ds)
plt.legend()
plt.title("Validation accuracy")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.savefig(os.path.join(working_dir, f"val_acc_{ts}.png"))
plt.close()
