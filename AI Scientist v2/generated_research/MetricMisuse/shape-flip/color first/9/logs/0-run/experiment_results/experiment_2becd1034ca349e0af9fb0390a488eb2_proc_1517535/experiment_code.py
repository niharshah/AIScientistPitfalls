import os, random, math, time, pathlib, itertools, numpy as np, torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ------------------------------------------------------------------ #
# mandatory working dir + device
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------ #
# experiment container
# ------------------------------------------------------------------ #
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


# ------------------------------------------------------------------ #
# metrics
# ------------------------------------------------------------------ #
def _uniq_colors(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _uniq_shapes(seq):
    return len({tok[0] for tok in seq.split() if tok})


def CWA(seqs, y, yhat):
    w = [_uniq_colors(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y, yhat) if t == p) / sum(w)


def SWA(seqs, y, yhat):
    w = [_uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y, yhat) if t == p) / sum(w)


def HWA(cwa, swa, eps=1e-9):
    return 2 * cwa * swa / (cwa + swa + eps)


# ------------------------------------------------------------------ #
# dataset loading (real or synthetic fallback)
# ------------------------------------------------------------------ #
def try_real():
    try:
        from SPR import load_spr_bench

        root = pathlib.Path("./SPR_BENCH")
        dset = load_spr_bench(root)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        print("Using synthetic data (couldn't load real):", e)
        return None


def synth(n=1000):
    shapes, colors = "ABCD", "1234"
    seq, lab = [], []
    for _ in range(n):
        L = random.randint(4, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
        s = " ".join(toks)
        seq.append(s)
        lab.append((_uniq_shapes(s) + _uniq_colors(s)) % 4)
    return {"sequence": seq, "label": lab}


real = try_real()
if real:
    train_raw, dev_raw, test_raw = real
else:
    train_raw, dev_raw, test_raw = synth(2000), synth(500), synth(500)


# ------------------------------------------------------------------ #
# vocabularies
# ------------------------------------------------------------------ #
def build_vocabs(ds_list):
    shapes, colors = set(), set()
    for ds in ds_list:
        seqs = ds["sequence"] if isinstance(ds, dict) else ds["sequence"]
        for s in seqs:
            for tok in s.split():
                shapes.add(tok[0])
                colors.add(tok[1])
    return {s: i for i, s in enumerate(sorted(shapes))}, {
        c: i for i, c in enumerate(sorted(colors))
    }


shape_vocab, color_vocab = build_vocabs([train_raw, dev_raw, test_raw])
num_shapes, num_colors = len(shape_vocab), len(color_vocab)


# ------------------------------------------------------------------ #
# sequence → PyG graph with 3 edge types
# ------------------------------------------------------------------ #
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    s_idx = [shape_vocab[t[0]] for t in toks]
    c_idx = [color_vocab[t[1]] for t in toks]
    pos = [i / (n - 1 if n > 1 else 1) for i in range(n)]

    # node features
    s_oh = torch.nn.functional.one_hot(torch.tensor(s_idx), num_classes=num_shapes)
    c_oh = torch.nn.functional.one_hot(torch.tensor(c_idx), num_classes=num_colors)
    pos_f = torch.tensor(pos, dtype=torch.float32).unsqueeze(1)
    x = torch.cat([s_oh.float(), c_oh.float(), pos_f], dim=1)

    # edges
    ei, et = [], []
    # sequential edges type 0
    for i in range(n - 1):
        ei += [[i, i + 1], [i + 1, i]]
        et += [0, 0]
    # same color edges type 1
    for i, j in itertools.combinations(range(n), 2):
        if c_idx[i] == c_idx[j]:
            ei += [[i, j], [j, i]]
            et += [1, 1]
    # same shape edges type 2
    for i, j in itertools.combinations(range(n), 2):
        if s_idx[i] == s_idx[j]:
            ei += [[i, j], [j, i]]
            et += [2, 2]
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(et, dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([int(label)]),
        seq=seq,
    )
    return data


def convert(raw):
    if isinstance(raw, dict):
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in raw]


train_ds, dev_ds, test_ds = map(convert, (train_raw, dev_raw, test_raw))
num_classes = len(set([d.y.item() for d in train_ds]))


# ------------------------------------------------------------------ #
# model
# ------------------------------------------------------------------ #
class RSPR(nn.Module):
    def __init__(self, in_dim, hid, classes, num_rel=3):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hid, num_rel)
        self.conv2 = RGCNConv(hid, hid, num_rel)
        self.lin = nn.Linear(hid, classes)

    def forward(self, data):
        x, ei, et, b = data.x, data.edge_index, data.edge_type, data.batch
        x = self.conv1(x, ei, et).relu()
        x = self.conv2(x, ei, et).relu()
        x = global_mean_pool(x, b)
        return self.lin(x)


# ------------------------------------------------------------------ #
# class weights for imbalance
# ------------------------------------------------------------------ #
cls_counts = torch.zeros(num_classes)
for d in train_ds:
    cls_counts[d.y.item()] += 1
cls_weights = (1.0 / cls_counts).to(device)


# ------------------------------------------------------------------ #
# one epoch
# ------------------------------------------------------------------ #
def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, seqs, ys, ps = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        if train:
            opt.zero_grad()
        out = model(batch)
        loss = crit(out, batch.y.view(-1))
        if train:
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(1).detach().cpu().tolist()
        labels = batch.y.view(-1).cpu().tolist()
        ps += preds
        ys += labels
        seqs += batch.seq
    avg_loss = tot_loss / len(loader.dataset)
    cwa, swa = CWA(seqs, ys, ps), SWA(seqs, ys, ps)
    return avg_loss, {"CWA": cwa, "SWA": swa, "HWA": HWA(cwa, swa)}, ys, ps


# ------------------------------------------------------------------ #
# training loop with lr sweep
# ------------------------------------------------------------------ #
EPOCHS = 12
batch_tr, batch_ev = 32, 64
LR_GRID = [5e-4, 1e-3]
crit = nn.CrossEntropyLoss(weight=cls_weights)

best_hwa, best_lr, best_state = -1, None, None

for lr in LR_GRID:
    print(f"\n--- LR={lr} ---")
    model = RSPR(num_shapes + num_colors + 1, 64, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader = DataLoader(train_ds, batch_tr, shuffle=True)
    val_loader = DataLoader(dev_ds, batch_ev)
    tr_losses = val_losses = []
    tr_metrics = val_metrics = []
    for ep in range(1, EPOCHS + 1):
        tl, tm, _, _ = run_epoch(model, tr_loader, crit, opt)
        vl, vm, _, _ = run_epoch(model, val_loader, crit)
        print(f"Epoch {ep}: validation_loss = {vl:.4f}")
        experiment_data["SPR"]["metrics"]["train"].append(tm)
        experiment_data["SPR"]["metrics"]["val"].append(vm)
        experiment_data["SPR"]["losses"]["train"].append(tl)
        experiment_data["SPR"]["losses"]["val"].append(vl)
    experiment_data["SPR"]["lr_values"].append(lr)
    if vm["HWA"] > best_hwa:
        best_hwa, best_lr = vm["HWA"], lr
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ------------------------------------------------------------------ #
# final test
# ------------------------------------------------------------------ #
print(f"\nBest LR = {best_lr}")
best_model = RSPR(num_shapes + num_colors + 1, 64, num_classes).to(device)
best_model.load_state_dict(best_state)
test_loader = DataLoader(test_ds, batch_ev)
_, tmet, gts, prs = run_epoch(best_model, test_loader, crit)
print(f"Test  CWA={tmet['CWA']:.4f}  SWA={tmet['SWA']:.4f}  HWA={tmet['HWA']:.4f}")

experiment_data["SPR"]["best_lr"] = best_lr
experiment_data["SPR"]["predictions"] = prs
experiment_data["SPR"]["ground_truth"] = gts
experiment_data["SPR"]["epochs"] = list(range(1, EPOCHS + 1))

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
