import os, random, math, time, pathlib, itertools
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt

# ---------------- house-keeping ------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- experiment data ----------------------
experiment_data = {"weight_decay": {}}  # will hold every run


# ---------------- metric helpers -----------------------
def _uniq_shapes(seq):
    return len(set(tok[0] for tok in seq.split()))


def _uniq_colors(seq):
    return len(set(tok[1] for tok in seq.split()))


def sdwa_metric(seqs, y_true, y_pred):
    weights = [_uniq_shapes(s) + _uniq_colors(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights)


# ---------------- data loading -------------------------
def try_load_real():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        return load_spr_bench(DATA_PATH)
    except Exception as e:
        print("Real dataset not found, generating synthetic data:", e)
        return None


def gen_synthetic(num):
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(num):
        length = random.randint(3, 10)
        tokens = [random.choice(shapes) + random.choice(colors) for _ in range(length)]
        seq = " ".join(tokens)
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


# ---------------- vocab build --------------------------
def build_vocabs(raw_sets):
    shapes, colors = set(), set()
    for rs in raw_sets:
        for seq in rs["sequence"]:
            for tok in seq.split():
                shapes.add(tok[0])
                colors.add(tok[1])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
    )


shape_vocab, color_vocab = build_vocabs([train_raw, dev_raw, test_raw])


# ---------------- graph conversion ---------------------
def seq_to_graph(seq, label):
    tokens = seq.split()
    n = len(tokens)
    shape_ids = [shape_vocab[t[0]] for t in tokens]
    color_ids = [color_vocab[t[1]] for t in tokens]
    pos_feat = [i / (n - 1 if n > 1 else 1) for i in range(n)]
    x = torch.tensor(
        [[sid, cid, pos] for sid, cid, pos in zip(shape_ids, color_ids, pos_feat)],
        dtype=torch.float,
    )
    edges = [[i, i + 1] for i in range(n - 1)]
    edge_index = torch.tensor(
        list(itertools.chain(edges, [(j, i) for i, j in edges])), dtype=torch.long
    ).t()
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), seq=seq
    )


def build_pyg_dataset(raw):
    if isinstance(raw, dict):
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in raw]


train_ds, dev_ds, test_ds = map(build_pyg_dataset, (train_raw, dev_raw, test_raw))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)


# ---------------- model definition ---------------------
class SPRGCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------------- training helpers ---------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, y_true, y_pred, seqs = 0, [], [], []
    for data in loader:
        data = data.to(device)
        if train:
            optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        if train:
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * data.num_graphs
        probs = out.argmax(1).cpu().tolist()
        y_pred.extend(probs)
        y_true.extend(data.y.cpu().tolist())
        seqs.extend(data.seq)
    return (
        tot_loss / len(loader.dataset),
        sdwa_metric(seqs, y_true, y_pred),
        y_true,
        y_pred,
    )


# ---------------- hyper-parameter sweep ----------------
EPOCHS = 10
weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2]

for wd in weight_decays:
    print("\n===== Training with weight_decay =", wd, "=====")
    run_key = str(wd)
    experiment_data["weight_decay"][run_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    model = SPRGCN(in_dim=3, hidden=64, out_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_sdwa, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_sdwa, _, _ = run_epoch(model, dev_loader)
        ed = experiment_data["weight_decay"][run_key]
        ed["epochs"].append(epoch)
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(tr_sdwa)
        ed["metrics"]["val"].append(val_sdwa)
        print(
            f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_SDWA={val_sdwa:.4f}"
        )

    # final test
    _, test_sdwa, gt, pr = run_epoch(model, test_loader)
    ed["predictions"] = pr
    ed["ground_truth"] = gt
    print(f"Test SDWA for wd={wd}: {test_sdwa:.4f}")

# ---------------- save artefacts -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ---------------- plotting -----------------------------
plt.figure()
for wd in weight_decays:
    key = str(wd)
    plt.plot(
        experiment_data["weight_decay"][key]["epochs"],
        experiment_data["weight_decay"][key]["losses"]["val"],
        label=f"wd={wd}",
    )
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.title("Weight-decay sweep")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"), dpi=150)
plt.close()
