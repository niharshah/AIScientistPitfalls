import os, random, math, time, pathlib, itertools, warnings
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- housekeeping ----------
warnings.filterwarnings("ignore")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- experiment dict ----------
experiment_data = {"batch_size": {}}  # will be filled with one key per batch size


# ---------- metric helpers ----------
def _uniq_shapes(seq):
    return len(set(tok[0] for tok in seq.split()))


def _uniq_colors(seq):
    return len(set(tok[1] for tok in seq.split()))


def sdwa_metric(seqs, y_true, y_pred):
    w = [_uniq_shapes(s) + _uniq_colors(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / sum(w)


# ---------- data loading (real or synthetic) ----------
def try_load_real():
    try:
        from SPR import load_spr_bench

        dpath = pathlib.Path("./SPR_BENCH")
        return load_spr_bench(dpath)
    except Exception as e:
        print("Real dataset not found, using synthetic.", e)
        return None


def gen_synthetic(n):
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(n):
        toks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(3, 10))
        ]
        seq = " ".join(toks)
        label = (_uniq_shapes(seq) * _uniq_colors(seq)) % 4
        seqs.append(seq)
        labels.append(label)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real["train"], real["dev"], real["test"]
else:
    train_raw, dev_raw, test_raw = (
        gen_synthetic(2000),
        gen_synthetic(500),
        gen_synthetic(500),
    )


# ---------- vocab ----------
def build_vocabs(rsets):
    shapes, colors = set(), set()
    for rs in rsets:
        for seq in rs["sequence"]:
            for tok in seq.split():
                shapes.add(tok[0])
                colors.add(tok[1])
    return {s: i for i, s in enumerate(sorted(shapes))}, {
        c: i for i, c in enumerate(sorted(colors))
    }


shape_vocab, color_vocab = build_vocabs([train_raw, dev_raw, test_raw])


# ---------- seq -> graph ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    sid = [shape_vocab[t[0]] for t in toks]
    cid = [color_vocab[t[1]] for t in toks]
    pos = [i / (n - 1 if n > 1 else 1) for i in range(n)]
    x = torch.tensor([[a, b, c] for a, b, c in zip(sid, cid, pos)], dtype=torch.float)
    edges = [[i, i + 1] for i in range(n - 1)]
    edge_ix = torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).t()
    return Data(x=x, edge_index=edge_ix, y=torch.tensor([label]), seq=seq)


def build_pyg_dataset(raw):
    if isinstance(raw, dict):
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in raw]


train_ds, dev_ds, test_ds = map(build_pyg_dataset, (train_raw, dev_raw, test_raw))


# ---------- model ----------
class SPRGCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = self.conv2(self.conv1(x, ei).relu(), ei).relu()
        return self.lin(global_mean_pool(x, b))


# ---------- training helpers ----------
def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, ys, ps, seqs = 0, [], [], []
    for d in loader:
        d = d.to(device)
        out = model(d)
        loss = criterion(out, d.y.view(-1))
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * d.num_graphs
        ps.extend(out.argmax(1).cpu().tolist())
        ys.extend(d.y.cpu().tolist())
        seqs.extend(d.seq)
    avg_loss = tot_loss / len(loader.dataset)
    return avg_loss, sdwa_metric(seqs, ys, ps), ys, ps


# ---------- hyper-parameter sweep ----------
BATCH_SIZES = [16, 32, 64, 128]
EPOCHS = 10

for bs in BATCH_SIZES:
    tag = f"{bs}"
    print(f"\n=== Training with batch_size={bs} ===")
    experiment_data["batch_size"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": list(range(1, EPOCHS + 1)),
    }
    # fresh model/optim
    model = SPRGCN(3, 64, 4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=max(64, bs))
    # epochs
    for ep in range(1, EPOCHS + 1):
        tr_l, tr_m, _, _ = run_epoch(model, train_loader, crit, optim)
        val_l, val_m, _, _ = run_epoch(model, dev_loader, crit)
        ed = experiment_data["batch_size"][tag]
        ed["losses"]["train"].append(tr_l)
        ed["losses"]["val"].append(val_l)
        ed["metrics"]["train"].append(tr_m)
        ed["metrics"]["val"].append(val_m)
        print(
            f"Epoch {ep:2d} | train_loss {tr_l:.3f} | val_loss {val_l:.3f} | val_SDWA {val_m:.3f}"
        )
    # final test
    test_loader = DataLoader(test_ds, batch_size=max(64, bs))
    _, tst_m, gt, pr = run_epoch(model, test_loader, crit)
    ed["predictions"], ed["ground_truth"] = pr, gt
    ed["test_SDWA"] = tst_m
    print(f"Test SDWA (bs={bs}): {tst_m:.3f}")
    # plot
    plt.figure()
    plt.plot(ed["epochs"], ed["losses"]["train"], label="train")
    plt.plot(ed["epochs"], ed["losses"]["val"], label="val")
    plt.title(f"Loss (batch={bs})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_bs_{bs}.png"), dpi=150)
    plt.close()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiments finished and saved.")
