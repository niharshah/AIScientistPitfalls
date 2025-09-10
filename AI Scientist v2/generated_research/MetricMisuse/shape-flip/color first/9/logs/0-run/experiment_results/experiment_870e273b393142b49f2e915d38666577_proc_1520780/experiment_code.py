import os, random, pathlib, math, time, itertools, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# --------------------------- I/O & misc ---------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "NoColorEmbedding": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
            "best_epoch": None,
        }
    }
}


# --------------------------- Metrics -------------------------------- #
def _uniq_colors(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _uniq_shapes(seq):
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y_true, y_pred):
    w = [_uniq_colors(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def swa(seqs, y_true, y_pred):
    w = [_uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def hwa(c, s, eps=1e-12):
    return 2 * c * s / (c + s + eps)


# --------------------------- Data ----------------------------------- #
def try_load_real():
    try:
        from SPR import load_spr_bench

        root = pathlib.Path("./SPR_BENCH")
        d = load_spr_bench(root)
        return d["train"], d["dev"], d["test"]
    except Exception as e:
        print("Could not load real SPR_BENCH:", e)
        return None


def gen_synth(n):
    sh, col = list("ABCD"), list("1234")
    seqs, labs = [], []
    for _ in range(n):
        ln = random.randint(3, 10)
        toks = [random.choice(sh) + random.choice(col) for _ in range(ln)]
        seq = " ".join(toks)
        lab = (len(set(t[0] for t in toks)) * len(set(t[1] for t in toks))) % 4
        seqs.append(seq)
        labs.append(lab)
    return {"sequence": seqs, "label": labs}


real = try_load_real()
train_raw, dev_raw, test_raw = (
    real if real else (gen_synth(2000), gen_synth(500), gen_synth(500))
)


# vocabularies
def build_vocabs(*splits):
    shapes, colors = set(), set()
    for split in splits:
        for s in split["sequence"] if isinstance(split, dict) else split["sequence"]:
            for tok in s.split():
                shapes.add(tok[0])
                colors.add(tok[1])
    return {s: i for i, s in enumerate(sorted(shapes))}, {
        c: i for i, c in enumerate(sorted(colors))
    }


shape_vocab, color_vocab = build_vocabs(train_raw, dev_raw, test_raw)
S, C = len(shape_vocab), len(color_vocab)


# --------------------------- Graph builders ------------------------- #
def seq_to_graph(seq, label, drop_color=False):
    toks = seq.split()
    n = len(toks)
    sid = [shape_vocab[t[0]] for t in toks]
    cid = [color_vocab[t[1]] for t in toks]
    pos = [i / (n - 1 if n > 1 else 1) for i in range(n)]
    sh_oh = torch.nn.functional.one_hot(torch.tensor(sid), num_classes=S).float()
    co_oh = torch.nn.functional.one_hot(torch.tensor(cid), num_classes=C).float()
    if drop_color:
        co_oh *= 0.0
    pos_feat = torch.tensor(pos, dtype=torch.float32).unsqueeze(1)
    x = torch.cat([sh_oh, co_oh, pos_feat], 1)

    edges = [(i, i + 1) for i in range(n - 1)]
    for s in set(sid):
        idx = [i for i, v in enumerate(sid) if v == s]
        edges += list(itertools.combinations(idx, 2))
    for c in set(cid):
        idx = [i for i, v in enumerate(cid) if v == c]
        edges += list(itertools.combinations(idx, 2))
    edges += [(j, i) for i, j in edges]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    return Data(x=x, edge_index=edge_index, y=torch.tensor([int(label)]), seq=seq)


def to_pyg(split, drop_color=False):
    if isinstance(split, dict):
        return [
            seq_to_graph(s, l, drop_color)
            for s, l in zip(split["sequence"], split["label"])
        ]
    else:
        return [
            seq_to_graph(ex["sequence"], int(ex["label"]), drop_color) for ex in split
        ]


train_ds, dev_ds, test_ds = map(
    lambda sp: to_pyg(sp, drop_color=True), (train_raw, dev_raw, test_raw)
)
num_classes = len({d.y.item() for d in train_ds + dev_ds + test_ds})


# --------------------------- Model ---------------------------------- #
class SPRGAT(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.g1 = GATConv(in_dim, hid, heads=4, concat=True, dropout=0.1)
        self.g2 = GATConv(hid * 4, hid, heads=4, concat=False, dropout=0.1)
        self.lin = nn.Linear(hid, out)

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = self.g1(x, ei).relu()
        x = self.g2(x, ei).relu()
        x = global_mean_pool(x, b)
        return self.lin(x)


# --------------------------- Training utils ------------------------- #
def run_epoch(model, loader, criterion, opt=None):
    model.train() if opt else model.eval()
    tot_loss, seqs, ys, ps = 0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        if opt:
            opt.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        if opt:
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(1).detach().cpu().tolist()
        labels = batch.y.view(-1).cpu().tolist()
        ps.extend(preds)
        ys.extend(labels)
        seqs.extend(batch.seq)
    avg_loss = tot_loss / len(loader.dataset)
    c, s = cwa(seqs, ys, ps), swa(seqs, ys, ps)
    return avg_loss, {"CWA": c, "SWA": s, "HWA": hwa(c, s)}, ys, ps


# --------------------------- Training loop -------------------------- #
BATCH, EPOCHS, LR = 32, 15, 5e-4
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(dev_ds, batch_size=2 * BATCH)
criterion = nn.CrossEntropyLoss()
model = SPRGAT(S + C + 1, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_hwa, best_state, best_ep = -1, None, 0
for epoch in range(1, EPOCHS + 1):
    tloss, tmet, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    vloss, vmet, _, _ = run_epoch(model, val_loader, criterion)
    print(f'Epoch {epoch}: val_loss={vloss:.4f} HWA={vmet["HWA"]:.4f}')
    d = experiment_data["NoColorEmbedding"]["SPR"]
    d["losses"]["train"].append(tloss)
    d["losses"]["val"].append(vloss)
    d["metrics"]["train"].append(tmet)
    d["metrics"]["val"].append(vmet)
    d["epochs"].append(epoch)
    if vmet["HWA"] > best_hwa:
        best_hwa, best_state, best_ep = (
            vmet["HWA"],
            {k: v.cpu() for k, v in model.state_dict().items()},
            epoch,
        )
        print(f"  New best at epoch {epoch} with HWA {best_hwa:.4f}")

# --------------------------- Test ----------------------------------- #
model.load_state_dict(best_state)
test_loader = DataLoader(test_ds, batch_size=64)
_, test_met, gt, pred = run_epoch(model, test_loader, criterion)
print(
    f'Test CWA={test_met["CWA"]:.3f} SWA={test_met["SWA"]:.3f} HWA={test_met["HWA"]:.3f}'
)

d = experiment_data["NoColorEmbedding"]["SPR"]
d["predictions"] = pred
d["ground_truth"] = gt
d["best_epoch"] = best_ep

# --------------------------- Save artefacts ------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure()
plt.plot(d["epochs"], d["losses"]["val"])
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve (NoColor)")
plt.savefig(os.path.join(working_dir, "val_loss.png"), dpi=150)
plt.close()
