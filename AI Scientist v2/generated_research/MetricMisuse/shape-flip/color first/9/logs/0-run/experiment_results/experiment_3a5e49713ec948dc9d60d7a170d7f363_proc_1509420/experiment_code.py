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
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt

# --------------- housekeeping & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------- experiment data container -------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# --------------- metric helpers ------------------------
def _uniq_shapes(seq):
    return len(set(tok[0] for tok in seq.split()))


def _uniq_colors(seq):
    return len(set(tok[1] for tok in seq.split()))


def sdwa_metric(seqs, y_true, y_pred):
    weights = [_uniq_shapes(s) + _uniq_colors(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights)


# --------------- data loading --------------------------
def try_load_real():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("./SPR_BENCH")  # adjust if needed
        dset = load_spr_bench(DATA_PATH)
        return dset
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
        label = (_uniq_shapes(seq) * _uniq_colors(seq)) % 4  # 4 classes
        seqs.append(seq)
        labels.append(label)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real["train"], real["dev"], real["test"]
else:
    train_raw = gen_synthetic(2000)
    dev_raw = gen_synthetic(500)
    test_raw = gen_synthetic(500)


# --------------- vocab build ---------------------------
def build_vocabs(raw_sets):
    shapes = set()
    colors = set()
    for rs in raw_sets:
        for seq in rs["sequence"]:
            for tok in seq.split():
                shapes.add(tok[0])
                colors.add(tok[1])
    shapes = sorted(list(shapes))
    colors = sorted(list(colors))
    return {s: i for i, s in enumerate(shapes)}, {c: i for i, c in enumerate(colors)}


shape_vocab, color_vocab = build_vocabs([train_raw, dev_raw, test_raw])


# --------------- graph conversion ----------------------
def seq_to_graph(seq, label):
    tokens = seq.split()
    n = len(tokens)
    shape_ids = [shape_vocab[t[0]] for t in tokens]
    color_ids = [color_vocab[t[1]] for t in tokens]
    pos_feat = [i / (n - 1 if n > 1 else 1) for i in range(n)]
    x = torch.tensor(
        [[sid, cid, pos] for sid, cid, pos in zip(shape_ids, color_ids, pos_feat)],
        dtype=torch.long if False else torch.float,
    )  # float features
    # edges: connect i<->i+1
    edges = [[i, i + 1] for i in range(n - 1)]
    edge_index = torch.tensor(
        list(itertools.chain(edges, [(j, i) for i, j in edges])), dtype=torch.long
    ).t()
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), seq=seq
    )


def build_pyg_dataset(raw):
    if isinstance(raw, dict):  # synthetic dict
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    else:  # HF dataset
        return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in raw]


train_ds = build_pyg_dataset(train_raw)
dev_ds = build_pyg_dataset(dev_raw)
test_ds = build_pyg_dataset(test_raw)


# --------------- model ---------------------------------
class SPRGCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


in_channels = 3  # shape id, color id, pos scalar
num_classes = 4
model = SPRGCN(in_channels, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# --------------- training loop -------------------------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, y_true, y_pred, seqs = 0, [], [], []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * data.num_graphs
        probs = out.argmax(dim=1).cpu().tolist()
        y_pred.extend(probs)
        y_true.extend(data.y.cpu().tolist())
        seqs.extend(data.seq)
    avg_loss = total_loss / len(loader.dataset)
    sdwa = sdwa_metric(seqs, y_true, y_pred)
    return avg_loss, sdwa, y_true, y_pred


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=64)

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_sdwa, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_sdwa, _, _ = run_epoch(dev_loader, train=False)
    experiment_data["SPR"]["epochs"].append(epoch)
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["train"].append(tr_sdwa)
    experiment_data["SPR"]["metrics"]["val"].append(val_sdwa)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_SDWA={val_sdwa:.4f}"
    )

# --------------- final test ----------------------------
test_loader = DataLoader(test_ds, batch_size=64)
_, test_sdwa, gt, pr = run_epoch(test_loader, train=False)
experiment_data["SPR"]["predictions"] = pr
experiment_data["SPR"]["ground_truth"] = gt
print(f"Test SDWA: {test_sdwa:.4f}")

# --------------- save artefacts ------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Loss curve plot
plt.figure()
plt.plot(
    experiment_data["SPR"]["epochs"],
    experiment_data["SPR"]["losses"]["train"],
    label="train",
)
plt.plot(
    experiment_data["SPR"]["epochs"],
    experiment_data["SPR"]["losses"]["val"],
    label="val",
)
plt.legend()
plt.title("GCN loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(working_dir, "loss_curve.png"), dpi=150)
plt.close()
