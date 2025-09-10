import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from datasets import load_dataset, DatasetDict

# ----- working dir -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----- metric helpers -----
def uniq_colors(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def uniq_shapes(seq):
    return len({tok[0] for tok in seq.split() if tok})


def color_weighted_accuracy(seqs, yt, yp):
    w = [uniq_colors(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, yt, yp) if t == p) / max(1, sum(w))


def shape_weighted_accuracy(seqs, yt, yp):
    w = [uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, yt, yp) if t == p) / max(1, sum(w))


def complexity_weighted_accuracy(seqs, yt, yp):
    w = [uniq_colors(s) * uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, yt, yp) if t == p) / max(1, sum(w))


# ----- dataset loading (fallback to synthetic) -----
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(n):
        return load_dataset(
            "csv", data_files=str(root / n), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {k: _ld(pathlib.Path(f"{k}.csv")) for k in ["train", "dev", "test"]}
    )


def get_dataset():
    try:
        root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        ds = load_spr_bench(root)
        print("Loaded SPR_BENCH from disk.")
    except Exception as e:
        print("Dataset not found, generating toy set.", e)
        shapes, colors = "ABC", "XYZ"

        def rand_seq():
            return " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(3, 8))
            )

        def make(n):
            return {
                "id": list(range(n)),
                "sequence": [rand_seq() for _ in range(n)],
                "label": [random.randint(0, 3) for _ in range(n)],
            }

        from datasets import Dataset

        ds = DatasetDict(
            {
                split: Dataset.from_dict(make(sz))
                for split, sz in [("train", 500), ("dev", 100), ("test", 100)]
            }
        )
    return ds


dset = get_dataset()

# ----- vocab -----
all_tokens = {
    tok for split in dset.values() for seq in split["sequence"] for tok in seq.split()
}
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id) + 1
num_classes = len(set(dset["train"]["label"]))


# ----- graph construction -----
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    x = torch.tensor([token2id[t] for t in toks], dtype=torch.long)
    edges = []
    # sequential
    edges += [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    # same color/shape
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i][1] == toks[j][1] or toks[i][0] == toks[j][0]:
                edges.append([i, j])
                edges.append([j, i])
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), seq=seq
    )


def build(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs, dev_graphs, test_graphs = map(
    build, (dset["train"], dset["dev"], dset["test"])
)
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)


# ----- model -----
class SPRGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.conv1 = GCNConv(64, 128)
        self.conv2 = GCNConv(128, 128)
        self.lin = nn.Linear(128, num_classes)

    def forward(self, data):
        x = self.emb(data.x.to(device))
        x = torch.relu(self.conv1(x, data.edge_index.to(device)))
        x = torch.relu(self.conv2(x, data.edge_index.to(device)))
        x = global_max_pool(x, data.batch.to(device))
        return self.lin(x)


# ----- training preparation -----
model = SPRGCN().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

epochs = 3
for epoch in range(1, epochs + 1):
    # train
    model.train()
    tloss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch.num_graphs
    tloss /= len(train_graphs)
    # validate
    model.eval()
    vloss = 0
    preds = []
    labels = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            vloss += loss.item() * batch.num_graphs
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(batch.y.view(-1).cpu().tolist())
            seqs.extend(batch.seq)
    vloss /= len(dev_graphs)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cpx = complexity_weighted_accuracy(seqs, labels, preds)
    ts = time.time()
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ts, tloss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ts, vloss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (ts, {"CWA": cwa, "SWA": swa, "CpxWA": cpx})
    )
    print(
        f"Epoch {epoch}: validation_loss = {vloss:.4f} | CWA {cwa:.3f} | SWA {swa:.3f} | CpxWA {cpx:.3f}"
    )

# ----- quick test evaluation -----
model.eval()
preds = []
labels = []
seqs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(batch.y.view(-1).cpu().tolist())
        seqs.extend(batch.seq)
print("Test CpxWA:", complexity_weighted_accuracy(seqs, labels, preds))

# ----- save all -----
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
