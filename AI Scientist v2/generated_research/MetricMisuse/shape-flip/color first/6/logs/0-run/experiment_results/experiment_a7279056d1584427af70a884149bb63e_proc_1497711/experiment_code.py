import os, pathlib, random, numpy as np, torch, matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ------------------- dirs & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- seeds -------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ------------------- metrics -----------------------
def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def color_weighted_accuracy(seqs: List[str], y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs: List[str], y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs: List[str], y_t, y_p):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


# ------------------- data (synth if real absent) ---
spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
have_real = spr_root.exists()


def load_real(name):
    from datasets import load_dataset

    return load_dataset(
        "csv",
        data_files=str(spr_root / f"{name}.csv"),
        split="train",
        cache_dir=".cache_dsets",
    )


def synth_rule_based(n: int) -> Dict[str, List]:
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(n):
        ln = np.random.randint(4, 9)
        tokens = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(tokens)
        s_var, c_var = count_shape_variety(seq), count_color_variety(seq)
        if s_var > c_var:
            lab = 0
        elif s_var == c_var:
            lab = 1
        else:
            lab = 2
        seqs.append(seq)
        labels.append(lab)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


if have_real:
    raw = {k: load_real(k) for k in ("train", "dev", "test")}
    print("Loaded real SPR_BENCH dataset.")
else:
    print("SPR_BENCH not found – generating synthetic data.")
    raw = {
        "train": synth_rule_based(5000),
        "dev": synth_rule_based(1000),
        "test": synth_rule_based(1000),
    }

# ------------------- vocab -------------------------
all_shapes, all_colors = set(), set()
for s in raw["train"]["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0])
        all_colors.add(tok[1])
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
num_shapes, num_colors = len(shape2idx), len(color2idx)
label_set = sorted(set(raw["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
num_class = len(label2idx)


# ------------------- graph builder -----------------
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    sh = torch.tensor([shape2idx[t[0]] for t in toks])
    col = torch.tensor([color2idx[t[1]] for t in toks])
    x = torch.stack([sh, col], 1)  # [n,2]

    src, dst, etype = [], [], []
    for i in range(n - 1):  # order edges
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [0, 0]
    for i in range(n):
        for j in range(i + 1, n):
            if int(col[i]) == int(col[j]):
                src += [i, j]
                dst += [j, i]
                etype += [1, 1]
            if int(sh[i]) == int(sh[j]):
                src += [i, j]
                dst += [j, i]
                etype += [2, 2]

    edge_index = torch.tensor([src, dst])
    edge_type = torch.tensor(etype)
    y = torch.tensor([label2idx[label]])
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


def build_dataset(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_ds, dev_ds, test_ds = map(build_dataset, (raw["train"], raw["dev"], raw["test"]))


# ------------------- model -------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, use_shape: bool = True, use_color: bool = True):
        super().__init__()
        self.use_shape, self.use_color = use_shape, use_color
        self.shape_emb = nn.Embedding(num_shapes, 8)
        self.color_emb = nn.Embedding(num_colors, 8)
        self.pre = nn.Linear(16, 32)
        self.conv1 = RGCNConv(32, 64, num_relations=3)
        self.conv2 = RGCNConv(64, 64, num_relations=3)
        self.cls = nn.Linear(64, num_class)

    def forward(self, data):
        sx = self.shape_emb(data.x[:, 0])
        cx = self.color_emb(data.x[:, 1])
        if not self.use_shape:
            sx = torch.zeros_like(sx)
        if not self.use_color:
            cx = torch.zeros_like(cx)
        x = torch.cat([sx, cx], 1)
        x = F.relu(self.pre(x))
        x = F.relu(self.conv1(x, data.edge_index, data.edge_type))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_type))
        x = global_mean_pool(x, data.batch)
        return self.cls(x)


# --------------- training / eval helpers -----------
def train_variant(name: str, use_shape: bool, use_color: bool, epochs: int = 6):
    model = SPR_RGCN(use_shape, use_color).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(train_ds, batch_size=64, shuffle=True)
    vl = DataLoader(dev_ds, batch_size=128, shuffle=False)
    e_data = {
        "metrics": {
            "train": {"CWA": [], "SWA": [], "CplxWA": []},
            "val": {"CWA": [], "SWA": [], "CplxWA": []},
        },
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        run_loss = 0.0
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * batch.num_graphs
        tr_loss = run_loss / len(tl.dataset)

        # ---- metrics on train ----
        model.eval()
        tr_seq, tr_t, tr_p = [], [], []
        with torch.no_grad():
            for batch in tl:
                batch = batch.to(device)
                o = model(batch)
                tr_p += o.argmax(1).cpu().tolist()
                tr_t += batch.y.cpu().tolist()
                tr_seq += batch.seq
        tr_cwa = color_weighted_accuracy(tr_seq, tr_t, tr_p)
        tr_swa = shape_weighted_accuracy(tr_seq, tr_t, tr_p)
        tr_cplx = complexity_weighted_accuracy(tr_seq, tr_t, tr_p)

        # ---- validation ----
        val_loss = 0.0
        v_seq, v_t, v_p = [], [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                o = model(batch)
                val_loss += crit(o, batch.y).item() * batch.num_graphs
                v_p += o.argmax(1).cpu().tolist()
                v_t += batch.y.cpu().tolist()
                v_seq += batch.seq
        val_loss /= len(vl.dataset)
        v_cwa = color_weighted_accuracy(v_seq, v_t, v_p)
        v_swa = shape_weighted_accuracy(v_seq, v_t, v_p)
        v_cplx = complexity_weighted_accuracy(v_seq, v_t, v_p)

        # ---- log ----
        e_data["epochs"].append(ep)
        e_data["losses"]["train"].append(tr_loss)
        e_data["losses"]["val"].append(val_loss)
        e_data["metrics"]["train"]["CWA"].append(tr_cwa)
        e_data["metrics"]["train"]["SWA"].append(tr_swa)
        e_data["metrics"]["train"]["CplxWA"].append(tr_cplx)
        e_data["metrics"]["val"]["CWA"].append(v_cwa)
        e_data["metrics"]["val"]["SWA"].append(v_swa)
        e_data["metrics"]["val"]["CplxWA"].append(v_cplx)
        print(f"[{name}] Ep{ep}: val_loss={val_loss:.4f} CplxWA={v_cplx:.4f}")
    # ---- test ----
    tl_test = DataLoader(test_ds, batch_size=128, shuffle=False)
    t_seq, t_t, t_p = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tl_test:
            batch = batch.to(device)
            o = model(batch)
            t_p += o.argmax(1).cpu().tolist()
            t_t += batch.y.cpu().tolist()
            t_seq += batch.seq
    test_cwa = color_weighted_accuracy(t_seq, t_t, t_p)
    test_swa = shape_weighted_accuracy(t_seq, t_t, t_p)
    test_cplx = complexity_weighted_accuracy(t_seq, t_t, t_p)
    e_data["predictions"] = t_p
    e_data["ground_truth"] = t_t
    e_data["metrics"]["test"] = {"CWA": test_cwa, "SWA": test_swa, "CplxWA": test_cplx}
    print(f"[{name}] Test CWA={test_cwa:.3f} SWA={test_swa:.3f} CplxWA={test_cplx:.3f}")

    # quick plots
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure()
    plt.plot(e_data["epochs"], e_data["losses"]["train"], label="train")
    plt.plot(e_data["epochs"], e_data["losses"]["val"], label="val")
    plt.title(f"{name} Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_{name}_{ts}.png"))
    plt.close()

    plt.figure()
    plt.plot(e_data["epochs"], e_data["metrics"]["val"]["CplxWA"])
    plt.title(f"{name} Val CplxWA")
    plt.xlabel("epoch")
    plt.ylabel("CplxWA")
    plt.savefig(os.path.join(working_dir, f"cplxwa_{name}_{ts}.png"))
    plt.close()
    return e_data


# ------------------- run all ablations --------------
experiment_data = {}
variants = [
    ("dual_channel", True, True),
    ("shape_only", True, False),
    ("color_only", False, True),
]
for name, use_s, use_c in variants:
    experiment_data[name] = {"SPR_BENCH": train_variant(name, use_s, use_c)}

# ------------------- save ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
