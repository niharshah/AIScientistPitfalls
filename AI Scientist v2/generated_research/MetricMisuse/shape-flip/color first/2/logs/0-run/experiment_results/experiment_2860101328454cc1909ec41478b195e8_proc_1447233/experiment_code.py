import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_max_pool
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weight(seq):
    return count_color_variety(seq) * count_shape_variety(seq)


def complexity_weighted_accuracy(seqs, y, p):
    w = [complexity_weight(s) for s in seqs]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- dataset loader ----------
def load_spr_bench(path):
    def _l(name):
        return load_dataset(
            "csv", data_files=str(path / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _l("train.csv")
    d["dev"] = _l("dev.csv")
    d["test"] = _l("test.csv")
    return d


def get_dataset():
    path_env = os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        d = load_spr_bench(pathlib.Path(path_env))
        print("Loaded SPR_BENCH from", path_env)
    except Exception as e:
        print("Dataset not found, generating synthetic:", e)
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

        d = DatasetDict()
        d["train"] = Dataset.from_dict(make(300))
        d["dev"] = Dataset.from_dict(make(60))
        d["test"] = Dataset.from_dict(make(60))
    return d


dset = get_dataset()

# ---------- vocab ----------
all_tokens = {
    tok for split in dset.values() for seq in split["sequence"] for tok in seq.split()
}
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id) + 1
num_classes = len(set(dset["train"]["label"]))


# ---------- graph construction ----------
def seq_to_graph(seq, lbl):
    toks = seq.split()
    n = len(toks)
    x = torch.tensor([token2id[t] for t in toks], dtype=torch.long)
    edge_src, edge_dst, edge_type = [], [], []

    # type 0: sequential edges
    for i in range(n - 1):
        edge_src.extend([i, i + 1])
        edge_dst.extend([i + 1, i])
        edge_type.extend([0, 0])

    # type 1: shared shape, type 2: shared color
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i][0] == toks[j][0]:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_type.extend([1, 1])
            if toks[i][1] == toks[j][1]:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_type.extend([2, 2])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([lbl], dtype=torch.long),
        seq=seq,
    )


def build(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs, dev_graphs = build(dset["train"]), build(dset["dev"])
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)


# ---------- model ----------
class RGCNClassifier(nn.Module):
    def __init__(self, vocab, nclass, hidden=128, num_relations=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.conv1 = RGCNConv(64, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)
        self.lin = nn.Linear(hidden, nclass)

    def forward(self, data):
        x = self.emb(data.x.to(device))
        x = torch.relu(
            self.conv1(x, data.edge_index.to(device), data.edge_type.to(device))
        )
        x = torch.relu(
            self.conv2(x, data.edge_index.to(device), data.edge_type.to(device))
        )
        x = global_max_pool(x, data.batch)
        return self.lin(x)


# ---------- training ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"val_CWA": [], "val_SWA": [], "val_CpxWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

model = RGCNClassifier(vocab_size, num_classes).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 5
for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    t_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * batch.num_graphs
    t_loss /= len(train_graphs)

    # ---- validate ----
    model.eval()
    v_loss = 0
    preds = []
    labels = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            v_loss += loss.item() * batch.num_graphs
            preds += out.argmax(1).cpu().tolist()
            labels += batch.y.view(-1).cpu().tolist()
            seqs += batch.seq
    v_loss /= len(dev_graphs)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cpx = complexity_weighted_accuracy(seqs, labels, preds)

    ts = time.time()
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ts, t_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ts, v_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val_CWA"].append((ts, cwa))
    experiment_data["SPR_BENCH"]["metrics"]["val_SWA"].append((ts, swa))
    experiment_data["SPR_BENCH"]["metrics"]["val_CpxWA"].append((ts, cpx))
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = labels

    print(
        f"Epoch {epoch}: validation_loss = {v_loss:.4f} | CWA {cwa:.4f} | SWA {swa:.4f} | CpxWA {cpx:.4f}"
    )

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All metrics saved to", os.path.join(working_dir, "experiment_data.npy"))
