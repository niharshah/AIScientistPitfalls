import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------- required boiler-plate ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers ----------
def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 0 if (cwa + swa) == 0 else 2 * cwa * swa / (cwa + swa)


# ---------- load SPR_BENCH or fallback ----------
def load_dataset_spr():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if root.exists():

        def _ld(csv):
            return load_dataset(
                "csv",
                data_files=str(root / csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = DatasetDict(
            {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
        )
        return d
    # synthetic tiny fallback
    shapes, colours = string.ascii_uppercase[:6], string.ascii_lowercase[:6]

    def _gen(n):
        seqs, lbl = [], []
        for _ in range(n):
            L = random.randint(4, 12)
            toks = [random.choice(shapes) + random.choice(colours) for _ in range(L)]
            seqs.append(" ".join(toks))
            lbl.append(int(toks[0][0] == toks[-1][0]))
        return {"sequence": seqs, "label": lbl}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(_gen(400)),
            "dev": Dataset.from_dict(_gen(120)),
            "test": Dataset.from_dict(_gen(200)),
        }
    )


dataset = load_dataset_spr()
num_classes = len(set(dataset["train"]["label"]))
print(
    f'Dataset sizes: train={len(dataset["train"])}, dev={len(dataset["dev"])}, test={len(dataset["test"])}'
)

# ---------- build vocabularies ----------
shape_vocab, colour_vocab = {}, {}


def idx(d, key):
    if key not in d:
        d[key] = len(d)
    return d[key]


max_len = 0
for seq in dataset["train"]["sequence"]:
    toks = seq.split()
    max_len = max(max_len, len(toks))
    for t in toks:
        idx(shape_vocab, t[0])
        idx(colour_vocab, t[1])


# ---------- graph construction ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shape_ids = torch.tensor([shape_vocab[t[0]] for t in toks], dtype=torch.long)
    colour_ids = torch.tensor([colour_vocab[t[1]] for t in toks], dtype=torch.long)
    pos_ids = torch.tensor(list(range(n)), dtype=torch.long)

    # sequential edges
    src = list(range(n - 1))
    dst = list(range(1, n))
    edges_s = [(i, j) for i, j in zip(src, dst)] + [(j, i) for i, j in zip(src, dst)]
    # same shape edges
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i][0] == toks[j][0]:
                edges_s.append((i, j))
                edges_s.append((j, i))
            if toks[i][1] == toks[j][1]:
                edges_s.append((i, j))
                edges_s.append((j, i))
    if edges_s:
        edge_index = torch.tensor(list(map(list, zip(*edges_s))), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        shape_id=shape_ids,
        colour_id=colour_ids,
        pos_id=pos_ids,
        seq_raw=seq,
    )
    return data


train_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["train"]["sequence"], dataset["train"]["label"])
]
dev_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["dev"]["sequence"], dataset["dev"]["label"])
]
test_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["test"]["sequence"], dataset["test"]["label"])
]


# ---------- GNN model ----------
class GNNClassifier(nn.Module):
    def __init__(self, s_vocab, c_vocab, max_pos, emb=32, hidden=64, num_cls=2):
        super().__init__()
        self.shape_emb = nn.Embedding(len(s_vocab), emb)
        self.col_emb = nn.Embedding(len(c_vocab), emb)
        self.pos_emb = nn.Embedding(max_pos + 1, emb)
        self.conv1 = SAGEConv(emb, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_cls)

    def forward(self, data):
        x = (
            self.shape_emb(data.shape_id)
            + self.col_emb(data.colour_id)
            + self.pos_emb(data.pos_id)
        )
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ---------- experiment tracking ----------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- training ----------
lr, epochs, bs_train, bs_eval = 1e-3, 8, 64, 128
model = GNNClassifier(
    shape_vocab, colour_vocab, max_len, emb=48, hidden=96, num_cls=num_classes
).to(device)
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_graphs, batch_size=bs_train, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=bs_eval)

for epoch in range(1, epochs + 1):
    # train
    model.train()
    tloss, tcorrect = 0.0, 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch.num_graphs
        tcorrect += (out.argmax(dim=-1) == batch.y).sum().item()
    tloss /= len(train_loader.dataset)
    tacc = tcorrect / len(train_loader.dataset)

    # validation
    model.eval()
    vloss, seqs, preds, gts = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            vloss += loss.item() * batch.num_graphs
            p = logits.argmax(dim=-1).cpu().tolist()
            g = batch.y.cpu().tolist()
            preds.extend(p)
            gts.extend(g)
            seqs.extend(batch.seq_raw)
    vloss /= len(dev_loader.dataset)
    vacc = float(np.mean([p == g for p, g in zip(preds, gts)]))
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hpa = harmonic_poly_accuracy(cwa, swa)

    # log
    experiment_data["SPR"]["losses"]["train"].append(tloss)
    experiment_data["SPR"]["losses"]["val"].append(vloss)
    experiment_data["SPR"]["metrics"]["train"].append({"acc": tacc})
    experiment_data["SPR"]["metrics"]["val"].append(
        {"acc": vacc, "cwa": cwa, "swa": swa, "hpa": hpa}
    )
    experiment_data["SPR"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: validation_loss = {vloss:.4f} | acc={vacc:.3f} CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}"
    )

experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment complete – data saved.")
