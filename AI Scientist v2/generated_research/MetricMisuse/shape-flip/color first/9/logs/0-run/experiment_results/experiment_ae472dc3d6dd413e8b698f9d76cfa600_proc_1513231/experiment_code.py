# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, math, time, pathlib, itertools
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# -------------------------------------------------- #
# mandatory working directory and device management
# -------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------- #
# global experiment container
# -------------------------------------------------- #
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},  # each entry will be dict(CWA,SWA,HM)
        "losses": {"train": [], "val": []},
        "lr_values": [],
        "epochs": [],
        "best_lr": None,
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------------------------------------- #
# Metric helpers (CWA, SWA, HM)
# -------------------------------------------------- #
def _uniq_colors(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _uniq_shapes(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y_true, y_pred):
    w = [_uniq_colors(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if w else 0.0


def swa(seqs, y_true, y_pred):
    w = [_uniq_shapes(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if w else 0.0


def hm_score(cwa_val, swa_val, eps=1e-12):
    return 2 * cwa_val * swa_val / (cwa_val + swa_val + eps)


# -------------------------------------------------- #
# Try to load real benchmark, else create synthetic
# -------------------------------------------------- #
def try_load_real():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        d = load_spr_bench(DATA_PATH)
        return d["train"], d["dev"], d["test"]
    except Exception as e:
        print("Real dataset not found, falling back to synthetic:", e)
        return None


def gen_synth(num=1000):
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(num):
        ln = random.randint(3, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(toks)
        label = (_uniq_shapes(seq) * _uniq_colors(seq)) % 4
        seqs.append(seq)
        labels.append(label)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real
else:
    train_raw, dev_raw, test_raw = gen_synth(2000), gen_synth(500), gen_synth(500)


# -------------------------------------------------- #
# Build vocabularies
# -------------------------------------------------- #
def build_vocabs(raw_sets):
    shapes, colors = set(), set()
    for rs in raw_sets:
        for s in rs["sequence"]:
            for tok in s.split():
                shapes.add(tok[0])
                colors.add(tok[1])
    return {s: i for i, s in enumerate(sorted(shapes))}, {
        c: i for i, c in enumerate(sorted(colors))
    }


shape_vocab, color_vocab = build_vocabs([train_raw, dev_raw, test_raw])
num_shapes, num_colors = len(shape_vocab), len(color_vocab)


# -------------------------------------------------- #
# Sequence -> PyG graph (with one-hot node features)
# -------------------------------------------------- #
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    s_ids = [shape_vocab[t[0]] for t in toks]
    c_ids = [color_vocab[t[1]] for t in toks]
    pos = [i / (n - 1 if n > 1 else 1) for i in range(n)]

    # one-hot encode and concat positional feature
    s_oh = torch.nn.functional.one_hot(torch.tensor(s_ids), num_classes=num_shapes)
    c_oh = torch.nn.functional.one_hot(torch.tensor(c_ids), num_classes=num_colors)
    pos_feat = torch.tensor(pos, dtype=torch.float32).unsqueeze(1)
    x = torch.cat(
        [s_oh.float(), c_oh.float(), pos_feat], dim=1
    )  # [n, num_shapes+num_colors+1]

    # simple chain edges (bi-directional)
    edges = [[i, i + 1] for i in range(n - 1)]
    edge_index = torch.tensor(edges + [[j, i] for i, j in edges], dtype=torch.long).t()
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([int(label)]), seq=seq)
    return data


def build_pyg(raw):
    if isinstance(raw, dict):
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    # hf Dataset case
    return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in raw]


train_ds, dev_ds, test_ds = map(build_pyg, (train_raw, dev_raw, test_raw))
num_classes = len(
    set(train_raw["label"] if isinstance(train_raw, dict) else train_raw["label"])
)


# -------------------------------------------------- #
# Model (architecture unchanged, new input dim)
# -------------------------------------------------- #
class SPRGCN(nn.Module):
    def __init__(self, in_dim, hid, classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = nn.Linear(hid, classes)

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = self.conv1(x, ei).relu()
        x = self.conv2(x, ei).relu()
        x = global_mean_pool(x, b)
        return self.lin(x)


# -------------------------------------------------- #
# Epoch runner
# -------------------------------------------------- #
def run_epoch(model, loader, criterion, opt=None):
    train_flag = opt is not None
    model.train() if train_flag else model.eval()
    total_loss, seqs, ys, ps = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        if train_flag:
            opt.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        if train_flag:
            loss.backward()
            opt.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(1).detach().cpu().tolist()
        labels = batch.y.view(-1).cpu().tolist()
        ps.extend(preds)
        ys.extend(labels)
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    cwa_val = cwa(seqs, ys, ps)
    swa_val = swa(seqs, ys, ps)
    hm_val = hm_score(cwa_val, swa_val)
    return avg_loss, {"CWA": cwa_val, "SWA": swa_val, "HM": hm_val}, ys, ps


# -------------------------------------------------- #
# Hyper-parameter sweep (learning rate)
# -------------------------------------------------- #
LR_GRID = [5e-4, 1e-4, 2e-3]
EPOCHS = 10
batch_train, batch_eval = 32, 64
criterion = nn.CrossEntropyLoss()

best_hm, best_lr, best_state = -1.0, None, None

for lr in LR_GRID:
    print(f"\n=== Training with lr={lr} ===")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    model = SPRGCN(num_shapes + num_colors + 1, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(dev_ds, batch_size=batch_eval)
    tr_losses = []
    val_losses = []
    tr_metrics = []
    val_metrics = []
    for ep in range(1, EPOCHS + 1):
        tl, tm, _, _ = run_epoch(model, tr_loader, criterion, optimizer)
        vl, vm, _, _ = run_epoch(model, val_loader, criterion)
        print(f"Epoch {ep:02d}: val_loss={vl:.4f}  HM={vm['HM']:.4f}")
        tr_losses.append(tl)
        val_losses.append(vl)
        tr_metrics.append(tm)
        val_metrics.append(vm)
    # log sweep data
    ed = experiment_data["SPR"]
    ed["lr_values"].append(lr)
    ed["losses"]["train"].append(tr_losses)
    ed["losses"]["val"].append(val_losses)
    ed["metrics"]["train"].append(tr_metrics)
    ed["metrics"]["val"].append(val_metrics)
    ed["epochs"] = list(range(1, EPOCHS + 1))
    # best check
    if val_metrics[-1]["HM"] > best_hm:
        best_hm, best_lr = val_metrics[-1]["HM"], lr
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"New best lr={lr} with HM={best_hm:.4f}")

# -------------------------------------------------- #
# Final evaluation on test split
# -------------------------------------------------- #
print(f"\nBest learning rate selected: {best_lr}")
best_model = SPRGCN(num_shapes + num_colors + 1, 64, num_classes).to(device)
best_model.load_state_dict(best_state)
test_loader = DataLoader(test_ds, batch_size=batch_eval)
_, test_metrics, gts, prs = run_epoch(best_model, test_loader, criterion)
print(
    f"Test metrics  CWA={test_metrics['CWA']:.4f}  "
    f"SWA={test_metrics['SWA']:.4f}  HM={test_metrics['HM']:.4f}"
)

# store predictions and gt
ed = experiment_data["SPR"]
ed["best_lr"] = best_lr
ed["predictions"] = prs
ed["ground_truth"] = gts

# -------------------------------------------------- #
# Save artefacts
# -------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot val loss curves
plt.figure()
for lr, vloss in zip(ed["lr_values"], ed["losses"]["val"]):
    plt.plot(ed["epochs"], vloss, label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("LR sweep - Validation loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "val_loss_lr_sweep.png"), dpi=150)
plt.close()
