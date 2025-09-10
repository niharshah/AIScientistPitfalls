import os, pathlib, numpy as np, torch, random, itertools
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset
from datetime import datetime

# ---------- working dir and device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- reproducibility ----------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# ---------- helpers ----------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) or 1.0)


# ---------- dataset load (real or synthetic) ----------
spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))


def load_csv(csv_name):
    return load_dataset(
        "csv",
        data_files=str(spr_root / csv_name),
        split="train",
        cache_dir=".cache_dsets",
    )


if spr_root.exists():
    print("Loading real SPR_BENCH from", spr_root)
    raw_train, raw_dev, raw_test = map(load_csv, ["train.csv", "dev.csv", "test.csv"])
else:
    print("SPR_BENCH not found. Building tiny synthetic fallback.")

    def make_synth(n):
        shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
        seqs, labels = [], []
        for i in range(n):
            length = np.random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            seqs.append(seq)
            labels.append(np.random.randint(0, 3))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    raw_train, raw_dev, raw_test = map(make_synth, [600, 150, 150])

# ---------- vocab ----------
all_shapes, all_colors = set(), set()
for s in raw_train["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0])
        all_colors.add(tok[1])
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
label2idx = {l: i for i, l in enumerate(sorted(set(raw_train["label"])))}
num_shapes, num_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)


# ---------- sequence → graph ----------
def seq_to_graph(seq: str, label: int):
    toks = seq.split()
    n = len(toks)
    x_np = np.array([[shape2idx[t[0]], color2idx[t[1]]] for t in toks], dtype=np.int64)
    edge_src, edge_dst, edge_types = [], [], []
    # type 0: adjacent position edges
    for i in range(n - 1):
        edge_src += [i, i + 1]
        edge_dst += [i + 1, i]
        edge_types += [0, 0]
    # type 1: same shape edges
    shape_map = {}
    for i, t in enumerate(toks):
        shape_map.setdefault(t[0], []).append(i)
    for idxs in shape_map.values():
        for i, j in itertools.combinations(idxs, 2):
            edge_src += [i, j]
            edge_dst += [j, i]
            edge_types += [1, 1]
    # type 2: same color edges
    color_map = {}
    for i, t in enumerate(toks):
        color_map.setdefault(t[1], []).append(i)
    for idxs in color_map.values():
        for i, j in itertools.combinations(idxs, 2):
            edge_src += [i, j]
            edge_dst += [j, i]
            edge_types += [2, 2]
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    data = Data(
        x=torch.tensor(x_np),
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label2idx[label]]),
        seq=seq,
    )
    return data


def build_dataset(raw_split):
    return [
        seq_to_graph(s, l) for s, l in zip(raw_split["sequence"], raw_split["label"])
    ]


train_set, dev_set, test_set = map(build_dataset, [raw_train, raw_dev, raw_test])


# ---------- model ----------
class SPR_RGCN(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, 8)
        self.color_emb = nn.Embedding(num_colors, 8)
        self.lin0 = nn.Linear(16, hid)
        self.rgcn1 = RGCNConv(hid, hid, num_relations=3)
        self.rgcn2 = RGCNConv(hid, hid, num_relations=3)
        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, batch):
        s_emb = self.shape_emb(batch.x[:, 0])
        c_emb = self.color_emb(batch.x[:, 1])
        x = torch.cat([s_emb, c_emb], dim=1)
        x = F.relu(self.lin0(x))
        x = F.relu(self.rgcn1(x, batch.edge_index, batch.edge_type))
        x = F.relu(self.rgcn2(x, batch.edge_index, batch.edge_type))
        x = global_mean_pool(x, batch.batch)
        return self.classifier(x)


# ---------- training config ----------
BATCH_SIZE = 64
EPOCHS = 6
experiment_data = {
    "spr_rgcn": {
        "metrics": {"train_compwa": [], "val_compwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

model = SPR_RGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- train / validate ----------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    avg_train_loss = total_loss / len(train_loader.dataset)

    # validation
    model.eval()
    val_loss, seqs, preds, gts = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            l = criterion(out, batch.y)
            val_loss += l.item() * batch.num_graphs
            preds += out.argmax(1).cpu().tolist()
            gts += batch.y.cpu().tolist()
            seqs += batch.seq
    avg_val_loss = val_loss / len(dev_loader.dataset)
    val_compwa = complexity_weighted_accuracy(seqs, gts, preds)

    # log
    ed = experiment_data["spr_rgcn"]
    ed["losses"]["train"].append(avg_train_loss)
    ed["losses"]["val"].append(avg_val_loss)
    ed["metrics"]["val_compwa"].append(val_compwa)
    ed["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} Val_CplxWA={val_compwa:.4f}"
    )

# ---------- test ----------
model.eval()
seqs, preds, gts = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds += out.argmax(1).cpu().tolist()
        gts += batch.y.cpu().tolist()
        seqs += batch.seq
test_compwa = complexity_weighted_accuracy(seqs, gts, preds)
experiment_data["spr_rgcn"]["predictions"] = preds
experiment_data["spr_rgcn"]["ground_truth"] = gts
experiment_data["spr_rgcn"]["metrics"]["test_compwa"] = test_compwa
print(f"Test Complexity-Weighted Accuracy: {test_compwa:.4f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
