import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# helper metric functions  (copied from prompt)
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(s) + count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ------------------------------------------------------------
# try to load real dataset; otherwise create synthetic one
def build_synthetic(num=512):
    shapes = list("ABCDE")
    colors = list("uvwxy")
    labels = ["rule1", "rule2"]
    rows = []
    for i in range(num):
        length = random.randint(4, 10)
        seq = " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )
        label = random.choice(labels)
        rows.append({"id": i, "sequence": seq, "label": label})
    return rows


def load_dataset_split(split):
    bench_path = pathlib.Path("SPR_BENCH")
    if bench_path.exists():
        import SPR  # uses the helper provided

        d = SPR.load_spr_bench(bench_path)[split]
        return [
            {"id": ex["id"], "sequence": ex["sequence"], "label": ex["label"]}
            for ex in d
        ]
    else:
        data = build_synthetic(20000 if split == "train" else 2500)
        return data


train_raw = load_dataset_split("train")
dev_raw = load_dataset_split("dev")
test_raw = load_dataset_split("test")
print(
    f"Loaded splits sizes -> train:{len(train_raw)}  dev:{len(dev_raw)}  test:{len(test_raw)}"
)

# ------------------------------------------------------------
# vocabularies
shape_set = set()
color_set = set()
label_set = set()
for ds in (train_raw, dev_raw, test_raw):
    for ex in ds:
        for tok in ex["sequence"].split():
            if len(tok) > 1:
                shape_set.add(tok[0])
                color_set.add(tok[1])
        label_set.add(ex["label"])
shape2id = {s: i for i, s in enumerate(sorted(shape_set))}
color2id = {c: i for i, c in enumerate(sorted(color_set))}
label2id = {l: i for i, l in enumerate(sorted(label_set))}
id2label = {i: l for l, i in label2id.items()}


# ------------------------------------------------------------
# convert to graph Data objects
def seq_to_graph(ex):
    tokens = ex["sequence"].split()
    n = len(tokens)
    # node features: [shape_id, color_id]
    feats = []
    for tok in tokens:
        s_id = shape2id[tok[0]]
        c_id = color2id[tok[1]]
        feats.append([s_id, c_id])
    x = torch.tensor(feats, dtype=torch.long)
    # edges
    if n == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2id[ex["label"]]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.seq_str = ex["sequence"]
    return data


train_data = [seq_to_graph(ex) for ex in train_raw]
dev_data = [seq_to_graph(ex) for ex in dev_raw]
test_data = [seq_to_graph(ex) for ex in test_raw]

# ------------------------------------------------------------
batch_size = 64
train_loader = GeoLoader(train_data, batch_size=batch_size, shuffle=True)
dev_loader = GeoLoader(dev_data, batch_size=batch_size)
test_loader = GeoLoader(test_data, batch_size=batch_size)


# ------------------------------------------------------------
# model
class SPR_GNN(nn.Module):
    def __init__(self, n_shape, n_color, n_label, emb_dim=32, hidden=64, layers=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.convs = nn.ModuleList()
        in_dim = emb_dim
        for _ in range(layers):
            self.convs.append(GCNConv(in_dim, hidden))
            in_dim = hidden
        self.act = nn.ReLU()
        self.lin = nn.Linear(hidden, n_label)

    def forward(self, data):
        shape_e = self.shape_emb(data.x[:, 0])
        color_e = self.color_emb(data.x[:, 1])
        h = shape_e + color_e
        for conv in self.convs:
            h = self.act(conv(h, data.edge_index))
        hg = global_mean_pool(h, data.batch)
        return self.lin(hg)


model = SPR_GNN(len(shape2id), len(color2id), len(label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    train_loss = total_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- eval
    model.eval()
    val_loss = 0
    all_preds, all_lbls, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            val_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1).cpu().tolist()
            lbls = batch.y.cpu().tolist()
            seqs = batch.seq_str
            all_preds.extend(preds)
            all_lbls.extend(lbls)
            all_seqs.extend(seqs)
    val_loss /= len(dev_loader.dataset)
    train_metric = None  # placeholder if needed
    val_cowa = complexity_weighted_accuracy(all_seqs, all_lbls, all_preds)

    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_cowa)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  CoWA = {val_cowa:.4f}")

# ------------------------------------------------------------
# final test evaluation
model.eval()
test_preds, test_lbls, test_seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        test_preds.extend(out.argmax(dim=1).cpu().tolist())
        test_lbls.extend(batch.y.cpu().tolist())
        test_seqs.extend(batch.seq_str)
test_cowa = complexity_weighted_accuracy(test_seqs, test_lbls, test_preds)
print(f"Test Complexity-Weighted Accuracy (CoWA): {test_cowa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_lbls
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
