import os, random, math, time, pathlib, itertools
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import matplotlib.pyplot as plt

# -------------------------------------------------- #
# mandatory working directory and device management
# -------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------- #
# experiment container
# -------------------------------------------------- #
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "lr_values": [],
        "best_lr": None,
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------------------------------------- #
# Metric helpers
# -------------------------------------------------- #
def _uniq_colors(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _uniq_shapes(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y_true, y_pred):
    w = [_uniq_colors(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1e-9)


def swa(seqs, y_true, y_pred):
    w = [_uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1e-9)


def hwa(cwa_v, swa_v, eps=1e-12):
    return 2 * cwa_v * swa_v / (cwa_v + swa_v + eps)


# -------------------------------------------------- #
# Try real dataset else synthetic
# -------------------------------------------------- #
def try_load_real():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        d = load_spr_bench(DATA_PATH)
        return d["train"], d["dev"], d["test"]
    except Exception as e:
        print("Real SPR_BENCH not found, using synthetic data instead.", e)
        return None


def gen_synth(n=1000):
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(n):
        ln = random.randint(3, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(toks)
        lab = (_uniq_shapes(seq) + _uniq_colors(seq)) % 4
        seqs.append(seq)
        labels.append(lab)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real
else:
    train_raw, dev_raw, test_raw = gen_synth(3000), gen_synth(600), gen_synth(600)


# -------------------------------------------------- #
# Build vocab
# -------------------------------------------------- #
def build_vocab(raw_sets):
    sset, cset = set(), set()
    for rs in raw_sets:
        data_iter = rs["sequence"] if isinstance(rs, dict) else rs["sequence"]
        for seq in data_iter:
            for tok in seq.split():
                sset.add(tok[0])
                cset.add(tok[1])
    return {s: i for i, s in enumerate(sorted(sset))}, {
        c: i for i, c in enumerate(sorted(cset))
    }


shape_vocab, color_vocab = build_vocab([train_raw, dev_raw, test_raw])
num_shapes, num_colors = len(shape_vocab), len(color_vocab)

# -------------------------------------------------- #
# Graph builder with relation edges
# -------------------------------------------------- #
REL_ADJ = 0
REL_COLOR = 1
REL_SHAPE = 2


def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    s_ids = [shape_vocab[t[0]] for t in toks]
    c_ids = [color_vocab[t[1]] for t in toks]
    pos = [i / (n - 1 if n > 1 else 1) for i in range(n)]

    # node features
    s_oh = torch.nn.functional.one_hot(torch.tensor(s_ids), num_classes=num_shapes)
    c_oh = torch.nn.functional.one_hot(torch.tensor(c_ids), num_classes=num_colors)
    pos_feat = torch.tensor(pos).unsqueeze(1)
    x = torch.cat([s_oh.float(), c_oh.float(), pos_feat], dim=1)

    edge_idx, edge_type = [], []

    # order edges
    for i in range(n - 1):
        edge_idx.append([i, i + 1])
        edge_idx.append([i + 1, i])
        edge_type += [REL_ADJ, REL_ADJ]
    # same color
    color2idx = dict()
    for idx, c in enumerate(c_ids):
        color2idx.setdefault(c, []).append(idx)
    for idxs in color2idx.values():
        for i, j in itertools.permutations(idxs, 2):
            edge_idx.append([i, j])
            edge_type.append(REL_COLOR)
    # same shape
    shape2idx = dict()
    for idx, s in enumerate(s_ids):
        shape2idx.setdefault(s, []).append(idx)
    for idxs in shape2idx.values():
        for i, j in itertools.permutations(idxs, 2):
            edge_idx.append([i, j])
            edge_type.append(REL_SHAPE)

    edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([int(label)]),
        seq=seq,
    )


def build_pyg(raw):
    if isinstance(raw, dict):
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    else:
        return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in raw]


train_ds, dev_ds, test_ds = map(build_pyg, (train_raw, dev_raw, test_raw))
num_classes = len(set([d.y.item() for d in train_ds]))


# -------------------------------------------------- #
# R-GCN model
# -------------------------------------------------- #
class SPR_RGCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_rel):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden, num_rel)
        self.conv2 = RGCNConv(hidden, hidden, num_rel)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, ei, et, b = data.x, data.edge_index, data.edge_type, data.batch
        x = self.conv1(x, ei, et).relu()
        x = self.conv2(x, ei, et).relu()
        x = global_mean_pool(x, b)
        return self.lin(x)


# -------------------------------------------------- #
# Epoch runner
# -------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, seqs, ys, preds = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad() if train_mode else None
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        pr = out.argmax(1).detach().cpu().tolist()
        gt = batch.y.view(-1).cpu().tolist()
        preds.extend(pr)
        ys.extend(gt)
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    cwa_v = cwa(seqs, ys, preds)
    swa_v = swa(seqs, ys, preds)
    hwa_v = hwa(cwa_v, swa_v)
    return avg_loss, {"CWA": cwa_v, "SWA": swa_v, "HWA": hwa_v}, ys, preds


# -------------------------------------------------- #
# Training with LR sweep
# -------------------------------------------------- #
LR_GRID = [5e-4, 1e-3, 2e-4]
EPOCHS = 12
batch_train, batch_eval = 32, 64
criterion = nn.CrossEntropyLoss()

best_hwa, best_lr, best_state = -1.0, None, None

for lr in LR_GRID:
    print(f"\n=== Training with lr={lr} ===")
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = SPR_RGCN(num_shapes + num_colors + 1, 64, num_classes, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(dev_ds, batch_size=batch_eval)
    tr_losses = []
    val_losses = []
    tr_ms = []
    val_ms = []
    for ep in range(1, EPOCHS + 1):
        tl, tm, _, _ = run_epoch(model, tr_loader, criterion, optimizer)
        vl, vm, _, _ = run_epoch(model, val_loader, criterion)
        print(f"Epoch {ep}: validation_loss = {vl:.4f}, HWA = {vm['HWA']:.4f}")
        tr_losses.append(tl)
        val_losses.append(vl)
        tr_ms.append(tm)
        val_ms.append(vm)
    ed = experiment_data["SPR"]
    ed["lr_values"].append(lr)
    ed["losses"]["train"].append(tr_losses)
    ed["losses"]["val"].append(val_losses)
    ed["metrics"]["train"].append(tr_ms)
    ed["metrics"]["val"].append(val_ms)
    ed["epochs"] = list(range(1, EPOCHS + 1))
    if val_ms[-1]["HWA"] > best_hwa:
        best_hwa = val_ms[-1]["HWA"]
        best_lr = lr
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"New best HWA {best_hwa:.4f} at lr {best_lr}")

# -------------------------------------------------- #
# Test evaluation
# -------------------------------------------------- #
print(f"\nBest LR selected: {best_lr}")
best_model = SPR_RGCN(num_shapes + num_colors + 1, 64, num_classes, 3).to(device)
best_model.load_state_dict(best_state)
test_loader = DataLoader(test_ds, batch_size=batch_eval)
_, test_metrics, gts, prs = run_epoch(best_model, test_loader, criterion)
print(
    f"Test CWA={test_metrics['CWA']:.4f}, SWA={test_metrics['SWA']:.4f}, HWA={test_metrics['HWA']:.4f}"
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

# Plot validation loss curves
plt.figure()
for lr, vloss in zip(ed["lr_values"], ed["losses"]["val"]):
    plt.plot(ed["epochs"], vloss, label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("R-GCN LR Sweep - Validation Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "val_loss_lr_sweep.png"), dpi=150)
plt.close()
