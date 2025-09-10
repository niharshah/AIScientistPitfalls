import os, random, itertools, pathlib, time, math, json

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np, torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


##############################################################################
# Helper: metrics
##############################################################################
def _uniq_colors(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _uniq_shapes(seq):
    return len({tok[0] for tok in seq.split() if tok})


def CWA(seqs, y_true, y_pred):
    w = [_uniq_colors(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def SWA(seqs, y_true, y_pred):
    w = [_uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def HWA(c, s, eps=1e-12):
    return 2 * c * s / (c + s + eps)


##############################################################################
# Load SPR_BENCH if available, else create synthetic fallback
##############################################################################
def try_load_real():
    try:
        from SPR import load_spr_bench

        root = pathlib.Path("./SPR_BENCH")
        dset = load_spr_bench(root)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        print("Could not load real SPR_BENCH:", e)
        return None


def make_synth(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        ln = random.randint(3, 12)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(toks)
        lab = (len(set(t[0] for t in toks)) * len(set(t[1] for t in toks))) % 4
        seqs.append(seq)
        labels.append(lab)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real
else:
    train_raw, dev_raw, test_raw = make_synth(3000), make_synth(600), make_synth(600)


##############################################################################
# Vocabularies
##############################################################################
def build_vocab(*splits):
    shapes, colors = set(), set()
    for split in splits:
        for seq in split["sequence"]:
            for t in seq.split():
                shapes.add(t[0])
                colors.add(t[1])
    return {s: i for i, s in enumerate(sorted(shapes))}, {
        c: i for i, c in enumerate(sorted(colors))
    }


shape_vocab, color_vocab = build_vocab(train_raw, dev_raw, test_raw)
S, C = len(shape_vocab), len(color_vocab)


##############################################################################
# Graph construction
##############################################################################
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shape_ids = [shape_vocab[t[0]] for t in toks]
    color_ids = [color_vocab[t[1]] for t in toks]
    pos_feat = torch.tensor(
        [i / (n - 1 if n > 1 else 1) for i in range(n)], dtype=torch.float32
    ).unsqueeze(1)

    edge_pairs, edge_types = [], []
    # order edges
    for i in range(n - 1):
        edge_pairs.extend([(i, i + 1), (i + 1, i)])
        edge_types.extend([0, 0])
    # same-shape
    for s in set(shape_ids):
        idx = [i for i, v in enumerate(shape_ids) if v == s]
        for i, j in itertools.combinations(idx, 2):
            edge_pairs.extend([(i, j), (j, i)])
            edge_types.extend([1, 1])
    # same-color
    for c in set(color_ids):
        idx = [i for i, v in enumerate(color_ids) if v == c]
        for i, j in itertools.combinations(idx, 2):
            edge_pairs.extend([(i, j), (j, i)])
            edge_types.extend([2, 2])

    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = (
        torch.tensor(edge_types, dtype=torch.long)
        if edge_types
        else torch.empty((0,), dtype=torch.long)
    )

    data = Data(
        x=pos_feat,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([int(label)]),
        shape_id=torch.tensor(shape_ids, dtype=torch.long),
        color_id=torch.tensor(color_ids, dtype=torch.long),
        seq=seq,
    )
    return data


def convert(split):
    if isinstance(split, dict):
        return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]
    else:
        return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in split]


train_ds, dev_ds, test_ds = map(convert, (train_raw, dev_raw, test_raw))
num_classes = len({d.y.item() for d in train_ds + dev_ds + test_ds})


##############################################################################
# Model
##############################################################################
class SPR_RGCN(nn.Module):
    def __init__(
        self, num_shapes, num_colors, emb_dim=16, hidden=64, num_rel=3, n_classes=4
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, emb_dim)
        self.color_emb = nn.Embedding(num_colors, emb_dim)
        in_dim = emb_dim * 2 + 1
        self.conv1 = RGCNConv(in_dim, hidden, num_rel)
        self.conv2 = RGCNConv(hidden, hidden, num_rel)
        self.lin = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        pos = data.x
        s_emb = self.shape_emb(data.shape_id)
        c_emb = self.color_emb(data.color_id)
        x = torch.cat([s_emb, c_emb, pos], dim=1)
        x = self.conv1(x, data.edge_index, data.edge_type).relu()
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index, data.edge_type).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


##############################################################################
# Training utilities
##############################################################################
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, seqs, ys, preds = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(1).detach().cpu().tolist()
        label = batch.y.view(-1).cpu().tolist()
        preds.extend(pred)
        ys.extend(label)
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    c, s = CWA(seqs, ys, preds), SWA(seqs, ys, preds)
    return avg_loss, {"CWA": c, "SWA": s, "HWA": HWA(c, s)}, ys, preds


##############################################################################
# Prepare loaders, model, optimizer
##############################################################################
BATCH = 32
EPOCHS = 20
LR = 5e-4
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(dev_ds, batch_size=2 * BATCH)
test_loader = DataLoader(test_ds, batch_size=2 * BATCH)

model = SPR_RGCN(S, C, n_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

##############################################################################
# Experiment data container
##############################################################################
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "best_epoch": None,
    }
}

##############################################################################
# Training loop with early best tracking
##############################################################################
best_hwa, best_state, best_ep = -1, None, 0
for epoch in range(1, EPOCHS + 1):
    tloss, tmet, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    vloss, vmet, _, _ = run_epoch(model, val_loader, criterion)
    print(f'Epoch {epoch}: validation_loss = {vloss:.4f}, HWA = {vmet["HWA"]:.4f}')
    experiment_data["SPR"]["losses"]["train"].append(tloss)
    experiment_data["SPR"]["losses"]["val"].append(vloss)
    experiment_data["SPR"]["metrics"]["train"].append(tmet)
    experiment_data["SPR"]["metrics"]["val"].append(vmet)
    experiment_data["SPR"]["epochs"].append(epoch)
    if vmet["HWA"] > best_hwa:
        best_hwa = vmet["HWA"]
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_ep = epoch
        print(f"  New best model at epoch {epoch} (HWA={best_hwa:.4f})")

##############################################################################
# Test evaluation
##############################################################################
model.load_state_dict(best_state)
test_loss, test_met, gt, pred = run_epoch(model, test_loader, criterion)
print(
    f'Test CWA={test_met["CWA"]:.3f} SWA={test_met["SWA"]:.3f} HWA={test_met["HWA"]:.3f}'
)

experiment_data["SPR"]["predictions"] = pred
experiment_data["SPR"]["ground_truth"] = gt
experiment_data["SPR"]["best_epoch"] = best_ep

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
