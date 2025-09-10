import os, pathlib, random, string, time, warnings, math, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers ----------
def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def CWA(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def SWA(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def HPA(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-9)


# ---------- dataset loading ----------
def load_or_synth():
    try:
        from datasets import load_dataset, DatasetDict

        root = pathlib.Path("./SPR_BENCH")
        if root.exists():

            def _ld(name):
                return load_dataset(
                    "csv",
                    data_files=str(root / name),
                    split="train",
                    cache_dir=".cache_dsets",
                )

            return DatasetDict(
                {
                    "train": _ld("train.csv"),
                    "dev": _ld("dev.csv"),
                    "test": _ld("test.csv"),
                }
            )
    except Exception as e:
        warnings.warn(str(e))
    # synthetic fallback
    shapes = list(string.ascii_uppercase[:6])
    colors = list(string.ascii_lowercase[:6])

    def make(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(4, 15)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seq = " ".join(toks)
            labels.append(int(toks[0][0] == toks[-1][0]))  # toy rule
            seqs.append(seq)
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(make(600)),
            "dev": Dataset.from_dict(make(200)),
            "test": Dataset.from_dict(make(300)),
        }
    )


ds = load_or_synth()
num_classes = len(set(ds["train"]["label"]))
print(
    f"dataset: {len(ds['train'])}/{len(ds['dev'])}/{len(ds['test'])}, classes={num_classes}"
)

# ---------- vocabulary ----------
vocab = {}


def add(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in ds["train"]["sequence"]:
    for t in seq.split():
        add(t)
vocab_size = len(vocab)
print("vocab:", vocab_size)


# ---------- graph builder ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    node_ids = torch.tensor([vocab[t] for t in toks], dtype=torch.long)
    edges = set()
    # sequential
    for i in range(n - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))
    # shape & color relations
    shape_map, color_map = {}, {}
    for i, t in enumerate(toks):
        shape_map.setdefault(t[0], []).append(i)
        color_map.setdefault(t[1], []).append(i)
    for lst in list(shape_map.values()) + list(color_map.values()):
        for i in lst:
            for j in lst:
                if i != j:
                    edges.add((i, j))
    if edges:
        ei = torch.tensor(list(zip(*edges)), dtype=torch.long)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
    data = Data(
        x=node_ids.unsqueeze(-1),
        edge_index=ei,
        y=torch.tensor([label], dtype=torch.long),
        seq_raw=seq,
    )
    return data


train_graphs = [
    seq_to_graph(s, l) for s, l in zip(ds["train"]["sequence"], ds["train"]["label"])
]
dev_graphs = [
    seq_to_graph(s, l) for s, l in zip(ds["dev"]["sequence"], ds["dev"]["label"])
]
test_graphs = [
    seq_to_graph(s, l) for s, l in zip(ds["test"]["sequence"], ds["test"]["label"])
]


# ---------- model ----------
class GNN(nn.Module):
    def __init__(self, vs, emb=32, hid=64, cls=2):
        super().__init__()
        self.emb = nn.Embedding(vs, emb)
        self.conv1 = SAGEConv(emb, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, cls)

    def forward(self, data):
        x = self.emb(data.x.squeeze()).to(device)
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
    }
}

# ---------- training ----------
lr = 1e-3
epochs = 10
bs_train = 32
bs_eval = 64
model = GNN(vocab_size, cls=num_classes).to(device)
opt = Adam(model.parameters(), lr=lr)
crit = nn.CrossEntropyLoss()

train_loader = DataLoader(train_graphs, batch_size=bs_train, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=bs_eval)
print("\nTraining...")
for epoch in range(1, epochs + 1):
    # --- train ---
    model.train()
    tloss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch)
        loss = crit(out, batch.y)
        loss.backward()
        opt.step()
        tloss += loss.item() * batch.num_graphs
    tloss /= len(train_loader.dataset)
    experiment_data["SPR"]["losses"]["train"].append(tloss)
    # --- val ---
    model.eval()
    vloss = 0.0
    preds = []
    gts = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = crit(logits, batch.y)
            vloss += loss.item() * batch.num_graphs
            p = logits.argmax(-1).cpu().tolist()
            g = batch.y.cpu().tolist()
            preds.extend(p)
            gts.extend(g)
            seqs.extend(batch.seq_raw)
    vloss /= len(dev_loader.dataset)
    experiment_data["SPR"]["losses"]["val"].append(vloss)
    acc = float(np.mean([p == t for p, t in zip(preds, gts)]))
    cwa = CWA(seqs, gts, preds)
    swa = SWA(seqs, gts, preds)
    hpa = HPA(cwa, swa)
    experiment_data["SPR"]["metrics"]["val"].append(
        {"acc": acc, "cwa": cwa, "swa": swa, "hpa": hpa}
    )
    print(
        f"Epoch {epoch}: val_loss={vloss:.4f}, acc={acc:.3f}, CWA={cwa:.3f}, SWA={swa:.3f}, HPA={hpa:.3f}"
    )

# ---------- final test evaluation ----------
test_loader = DataLoader(test_graphs, batch_size=bs_eval)
model.eval()
preds = []
gts = []
seqs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq_raw)
cwa = CWA(seqs, gts, preds)
swa = SWA(seqs, gts, preds)
hpa = HPA(cwa, swa)
print(f"\nTEST  CWA={cwa:.3f}, SWA={swa:.3f}, HPA={hpa:.3f}")

experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
