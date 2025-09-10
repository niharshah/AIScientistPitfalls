import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from datasets import load_dataset

# ---------- work dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers ----------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


def cplxwa(seqs, y_t, y_p):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


# ---------- load dataset (real or synthetic) ----------
spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))


def _load_csv(split_csv):
    return load_dataset(
        "csv",
        data_files=str(spr_root / split_csv),
        split="train",
        cache_dir=".cache_dsets",
    )


if spr_root.exists():
    print("Loading real SPR_BENCH ...")
    raw_train, raw_dev, raw_test = [
        _load_csv(f"{sp}.csv") for sp in ("train", "dev", "test")
    ]
else:
    print("SPR_BENCH not found. Generating synthetic toy data.")

    def make_synth(n):
        shapes, colors = "ABC", "123"
        seqs, labels = [], []
        rng = np.random.default_rng(0)
        for i in range(n):
            length = rng.integers(4, 9)
            seq = " ".join(
                rng.choice(list(shapes)) + rng.choice(list(colors))
                for _ in range(length)
            )
            seqs.append(seq)
            labels.append(rng.integers(0, 3))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    raw_train, raw_dev, raw_test = map(make_synth, (600, 150, 300))

# ---------- vocab ----------
all_shapes, all_colors = set(), set()
for s in raw_train["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0])
        all_colors.add(tok[1])
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
num_shapes, num_colors = len(shape2idx), len(color2idx)
num_classes = len(set(raw_train["label"]))


# ---------- graph builder ----------
def seq_to_graph(seq, label):
    toks = seq.strip().split()
    n = len(toks)
    x = torch.tensor(
        [[shape2idx[t[0]], color2idx[t[1]]] for t in toks], dtype=torch.long
    )
    # edges: adjacency + same-shape + same-color
    edges = set()
    for i in range(n - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i][0] == toks[j][0] or toks[i][1] == toks[j][1]:
                edges.add((i, j))
                edges.add((j, i))
    if edges:
        ei = torch.tensor(list(zip(*edges)), dtype=torch.long)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=ei, y=y, seq=seq)


def build_dataset(raw):
    if isinstance(raw, dict):  # synthetic
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    return [seq_to_graph(r["sequence"], r["label"]) for r in raw]


train_data, dev_data, test_data = map(build_dataset, (raw_train, raw_dev, raw_test))


# ---------- model ----------
class SPRGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, 8)
        self.color_emb = nn.Embedding(num_colors, 8)
        self.pre_lin = nn.Linear(16, 32)
        self.conv1 = GraphConv(32, 64)
        self.conv2 = GraphConv(64, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, data):
        x = torch.cat([self.shape_emb(data.x[:, 0]), self.color_emb(data.x[:, 1])], 1)
        x = F.relu(self.pre_lin(x))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)


# ---------- experiment data dict ----------
experiment_data = {
    "SPR": {
        "metrics": {"train_cplxwa": [], "val_cplxwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- training setup ----------
batch_size, epochs = 64, 3
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=128, shuffle=False)
model = SPRGNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- training loop ----------
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.num_graphs
    avg_train_loss = running_loss / len(train_loader.dataset)
    # compute train cplxwa quickly using a single pass (optional)
    model.eval()
    tr_seq, tr_true, tr_pred = [], [], []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            pr = model(batch).argmax(1).cpu().tolist()
            tr_pred.extend(pr)
            tr_true.extend(batch.y.cpu().tolist())
            tr_seq.extend(batch.seq)
    train_cplx = cplxwa(tr_seq, tr_true, tr_pred)
    # validation
    val_loss = 0
    val_seq, val_true, val_pred = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            val_loss += criterion(out, batch.y).item() * batch.num_graphs
            val_pred.extend(out.argmax(1).cpu().tolist())
            val_true.extend(batch.y.cpu().tolist())
            val_seq.extend(batch.seq)
    avg_val_loss = val_loss / len(dev_loader.dataset)
    val_cplx = cplxwa(val_seq, val_true, val_pred)
    # log
    experiment_data["SPR"]["losses"]["train"].append(avg_train_loss)
    experiment_data["SPR"]["losses"]["val"].append(avg_val_loss)
    experiment_data["SPR"]["metrics"]["train_cplxwa"].append(train_cplx)
    experiment_data["SPR"]["metrics"]["val_cplxwa"].append(val_cplx)
    experiment_data["SPR"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
        f"train_CplxWA={train_cplx:.4f} val_CplxWA={val_cplx:.4f}"
    )

# ---------- test evaluation ----------
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
model.eval()
ts_seq, ts_true, ts_pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        ts_pred.extend(out.argmax(1).cpu().tolist())
        ts_true.extend(batch.y.cpu().tolist())
        ts_seq.extend(batch.seq)
test_cplx = cplxwa(ts_seq, ts_true, ts_pred)
test_cwa = cwa(ts_seq, ts_true, ts_pred)
test_swa = swa(ts_seq, ts_true, ts_pred)
experiment_data["SPR"]["predictions"] = ts_pred
experiment_data["SPR"]["ground_truth"] = ts_true
experiment_data["SPR"]["metrics"]["test_cplxwa"] = test_cplx
experiment_data["SPR"]["metrics"]["test_cwa"] = test_cwa
experiment_data["SPR"]["metrics"]["test_swa"] = test_swa
print(f"TEST  CplxWA={test_cplx:.4f}  CWA={test_cwa:.4f}  SWA={test_swa:.4f}")

# ---------- plots ----------
epochs_list = experiment_data["SPR"]["epochs"]
plt.figure()
plt.plot(epochs_list, experiment_data["SPR"]["losses"]["train"], label="train")
plt.plot(epochs_list, experiment_data["SPR"]["losses"]["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves")
plt.savefig(os.path.join(working_dir, "loss_curves.png"))
plt.close()

plt.figure()
plt.plot(epochs_list, experiment_data["SPR"]["metrics"]["train_cplxwa"], label="train")
plt.plot(epochs_list, experiment_data["SPR"]["metrics"]["val_cplxwa"], label="val")
plt.xlabel("Epoch")
plt.ylabel("CplxWA")
plt.legend()
plt.title("CplxWA curves")
plt.savefig(os.path.join(working_dir, "cplxwa_curves.png"))
plt.close()

# ---------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to ./working/experiment_data.npy")
