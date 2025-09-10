import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINEConv, global_max_pool
from datasets import load_dataset, DatasetDict

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers ----------
def count_color_variety(sequence):  # token = ShapeChar + ColorChar
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y, p):  # CpxWA
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- dataset loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def get_dataset():
    default_path = "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
    try:
        d = load_spr_bench(pathlib.Path(default_path))
        print(f"Loaded SPR_BENCH from {default_path}")
    except Exception as e:
        # Fallback synthetic toy set
        print("Dataset not found, using small synthetic set", e)
        shapes, colors = "ABC", "XYZ"

        def rand_seq():
            return " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 8))
            )

        def make(n):
            return {
                "id": list(range(n)),
                "sequence": [rand_seq() for _ in range(n)],
                "label": [random.randint(0, 3) for _ in range(n)],
            }

        from datasets import Dataset

        d = DatasetDict(
            train=Dataset.from_dict(make(300)),
            dev=Dataset.from_dict(make(60)),
            test=Dataset.from_dict(make(60)),
        )
    return d


dset = get_dataset()

# ---------- vocab ----------
all_tokens = {
    tok for split in dset.values() for seq in split["sequence"] for tok in seq.split()
}
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id) + 1


# ---------- graph construction ----------
def build_edges(tokens):
    n = len(tokens)
    edges, attrs = [], []
    colors = [tok[1] if len(tok) > 1 else "" for tok in tokens]
    shapes = [tok[0] for tok in tokens]
    # sequential edges (type 0)
    for i in range(n - 1):
        for a, b in ((i, i + 1), (i + 1, i)):
            edges.append((a, b))
            attrs.append([1, 0, 0])
    # same color edges (type 1)
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                edges.extend([(i, j), (j, i)])
                attrs.extend([[0, 1, 0], [0, 1, 0]])
    # same shape edges (type 2)
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i] == shapes[j]:
                edges.extend([(i, j), (j, i)])
                attrs.extend([[0, 0, 1], [0, 0, 1]])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ea = torch.tensor(attrs, dtype=torch.float)
    return ei, ea


def seq_to_graph(seq, label):
    toks = seq.split()
    x = torch.tensor([token2id[t] for t in toks], dtype=torch.long)
    edge_index, edge_attr = build_edges(toks)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        seq=seq,
    )


def build_graph_list(split):  # returns list[Data]
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs, dev_graphs, test_graphs = map(
    build_graph_list, (dset["train"], dset["dev"], dset["test"])
)
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)


# ---------- model ----------
class SPR_GNN(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        hid = 64
        self.node_emb = nn.Embedding(vocab_size, hid)
        self.edge_encoder = nn.Linear(3, hid)  # 3 relation types
        nn1 = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
        nn2 = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn2)
        self.classifier = nn.Linear(hid, n_classes)

    def forward(self, data):
        x = self.node_emb(data.x)
        e = self.edge_encoder(data.edge_attr)
        x = torch.relu(self.conv1(x, data.edge_index, e))
        x = torch.relu(self.conv2(x, data.edge_index, e))
        x = global_max_pool(x, data.batch)
        return self.classifier(x)


num_classes = len(set(dset["train"]["label"]))
model = SPR_GNN(vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "CWA": {"train": [], "val": []},
            "SWA": {"train": [], "val": []},
            "CpxWA": {"train": [], "val": []},
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- training loop ----------
epochs = 8
for epoch in range(1, epochs + 1):
    # ---------- train ----------
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.num_graphs
    train_loss = running_loss / len(train_graphs)
    # ---------- validation ----------
    model.eval()
    vloss, seqs, preds, labels = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            vloss += loss.item() * batch.num_graphs
            preds += out.argmax(1).cpu().tolist()
            labels += batch.y.view(-1).cpu().tolist()
            seqs += batch.seq
    val_loss = vloss / len(dev_graphs)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cpx = complexity_weighted_accuracy(seqs, labels, preds)
    ts = time.time()
    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ts, train_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ts, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["CWA"]["val"].append((ts, cwa))
    experiment_data["SPR_BENCH"]["metrics"]["SWA"]["val"].append((ts, swa))
    experiment_data["SPR_BENCH"]["metrics"]["CpxWA"]["val"].append((ts, cpx))
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"CWA={cwa:.4f}  SWA={swa:.4f}  CpxWA={cpx:.4f}"
    )

# ---------- test evaluation ----------
model.eval()
seqs, preds, labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds += out.argmax(1).cpu().tolist()
        labels += batch.y.view(-1).cpu().tolist()
        seqs += batch.seq
cwa_test = color_weighted_accuracy(seqs, labels, preds)
swa_test = shape_weighted_accuracy(seqs, labels, preds)
cpx_test = complexity_weighted_accuracy(seqs, labels, preds)
print(
    "\nTest CWA {:.4f} | SWA {:.4f} | CpxWA {:.4f}".format(cwa_test, swa_test, cpx_test)
)
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
