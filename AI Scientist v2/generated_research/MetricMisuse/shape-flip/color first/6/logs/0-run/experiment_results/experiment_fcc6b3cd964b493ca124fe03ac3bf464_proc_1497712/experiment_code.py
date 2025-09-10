import os, pathlib, random, numpy as np, torch, matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv  # no mean pool this time

# ------------------- experiment dict -------------------
experiment_data = {
    "pooling_last_token": {
        "SPR_BENCH": {
            "metrics": {
                "train": {"CWA": [], "SWA": [], "CplxWA": []},
                "val": {"CWA": [], "SWA": [], "CplxWA": []},
            },
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------- working dir -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- device ------------------------
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


# ------------------- data loading ------------------
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
    """Generate sequences + rule-based label:
    label 0: uniq_shapes  >  uniq_colors
    label 1: uniq_shapes == uniq_colors
    label 2: uniq_shapes  <  uniq_colors
    """
    shapes = list("ABCD")
    colors = list("1234")
    seqs, labels = [], []
    for _ in range(n):
        ln = np.random.randint(4, 9)
        tokens = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(tokens)
        s_var = count_shape_variety(seq)
        c_var = count_color_variety(seq)
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
    raw = {
        "train": load_real("train"),
        "dev": load_real("dev"),
        "test": load_real("test"),
    }
    print("Loaded real SPR_BENCH dataset.")
else:
    print("SPR_BENCH folder not found – synthesising rule-based data.")
    raw = {
        "train": synth_rule_based(8000),
        "dev": synth_rule_based(2000),
        "test": synth_rule_based(2000),
    }

# ------------------- vocab building ----------------
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
    sh = torch.tensor([shape2idx[t[0]] for t in toks], dtype=torch.long)
    col = torch.tensor([color2idx[t[1]] for t in toks], dtype=torch.long)
    x = torch.stack([sh, col], 1)  # [n, 2]

    edges_src, edges_dst, etype = [], [], []

    # 0: order edges (i -> i+1)
    for i in range(n - 1):
        edges_src += [i, i + 1]
        edges_dst += [i + 1, i]
        etype += [0, 0]

    # 1: same color, 2: same shape
    for i in range(n):
        for j in range(i + 1, n):
            if int(col[i]) == int(col[j]):
                edges_src += [i, j]
                edges_dst += [j, i]
                etype += [1, 1]
            if int(sh[i]) == int(sh[j]):
                edges_src += [i, j]
                edges_dst += [j, i]
                etype += [2, 2]

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


def build_dataset(split):
    if have_real:
        return [seq_to_graph(rec["sequence"], rec["label"]) for rec in split]
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_ds, dev_ds, test_ds = map(build_dataset, (raw["train"], raw["dev"], raw["test"]))


# ------------------- model -------------------------
class SPR_RGCN_LastToken(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, 8)
        self.color_emb = nn.Embedding(num_colors, 8)
        self.pre = nn.Linear(16, 32)
        self.conv1 = RGCNConv(32, 64, num_relations=3)
        self.conv2 = RGCNConv(64, 64, num_relations=3)
        self.cls = nn.Linear(64, num_class)

    def forward(self, data):
        sx = self.shape_emb(data.x[:, 0])
        cx = self.color_emb(data.x[:, 1])
        x = torch.cat([sx, cx], 1)
        x = F.relu(self.pre(x))
        x = F.relu(self.conv1(x, data.edge_index, data.edge_type))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_type))

        # indices of last node for each graph
        last_idx = data.ptr[1:] - 1  # ptr len = batch_size+1
        x = x[last_idx]  # [batch_size, feat_dim]
        return self.cls(x)


model = SPR_RGCN_LastToken().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------- loaders -----------------------
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# ------------------- training loop -----------------
epochs = 10
ed = experiment_data["pooling_last_token"]["SPR_BENCH"]
for epoch in range(1, epochs + 1):
    # ---- training ----
    model.train()
    run_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch.num_graphs
    tr_loss = run_loss / len(train_loader.dataset)

    # compute train metrics
    model.eval()
    tr_seq, tr_true, tr_pred = [], [], []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            tr_pred.extend(out.argmax(1).cpu().tolist())
            tr_true.extend(batch.y.cpu().tolist())
            tr_seq.extend(batch.seq)
    tr_cwa = color_weighted_accuracy(tr_seq, tr_true, tr_pred)
    tr_swa = shape_weighted_accuracy(tr_seq, tr_true, tr_pred)
    tr_cplx = complexity_weighted_accuracy(tr_seq, tr_true, tr_pred)

    # ---- validation ----
    val_loss, v_seq, v_true, v_pred = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            val_loss += criterion(out, batch.y).item() * batch.num_graphs
            v_pred.extend(out.argmax(1).cpu().tolist())
            v_true.extend(batch.y.cpu().tolist())
            v_seq.extend(batch.seq)
    val_loss /= len(dev_loader.dataset)
    val_cwa = color_weighted_accuracy(v_seq, v_true, v_pred)
    val_swa = shape_weighted_accuracy(v_seq, v_true, v_pred)
    val_cplx = complexity_weighted_accuracy(v_seq, v_true, v_pred)

    # ---- logging ----
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"]["CWA"].append(tr_cwa)
    ed["metrics"]["train"]["SWA"].append(tr_swa)
    ed["metrics"]["train"]["CplxWA"].append(tr_cplx)
    ed["metrics"]["val"]["CWA"].append(val_cwa)
    ed["metrics"]["val"]["SWA"].append(val_swa)
    ed["metrics"]["val"]["CplxWA"].append(val_cplx)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}  "
        f"CplxWA={val_cplx:.4f}  CWA={val_cwa:.4f}  SWA={val_swa:.4f}"
    )

# ------------------- test evaluation ----------------
model.eval()
t_seq, t_true, t_pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        t_pred.extend(out.argmax(1).cpu().tolist())
        t_true.extend(batch.y.cpu().tolist())
        t_seq.extend(batch.seq)

test_cwa = color_weighted_accuracy(t_seq, t_true, t_pred)
test_swa = shape_weighted_accuracy(t_seq, t_true, t_pred)
test_cplx = complexity_weighted_accuracy(t_seq, t_true, t_pred)
print(f"Test CWA={test_cwa:.3f}  SWA={test_swa:.3f}  CplxWA={test_cplx:.3f}")

ed["predictions"] = t_pred
ed["ground_truth"] = t_true
ed["metrics"]["test"] = {"CWA": test_cwa, "SWA": test_swa, "CplxWA": test_cplx}

# ------------------- save results ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ------------------- quick plots -------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure()
plt.plot(ed["epochs"], ed["losses"]["train"], label="train")
plt.plot(ed["epochs"], ed["losses"]["val"], label="val")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss (Last-token pooling)")
plt.legend()
plt.savefig(os.path.join(working_dir, f"loss_lastpool_{ts}.png"))
plt.close()

plt.figure()
plt.plot(ed["epochs"], ed["metrics"]["val"]["CplxWA"])
plt.xlabel("epoch")
plt.ylabel("CplxWA")
plt.title("Validation CplxWA (Last-token pooling)")
plt.savefig(os.path.join(working_dir, f"cplxwa_lastpool_{ts}.png"))
plt.close()
