import os, pathlib, random, string, time, math, json, warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool

# ----------------- working directory & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- metric helpers --------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ------------------------ data utils --------------------------
def try_load_benchmark():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    dset = DatasetDict()
    dset["train"], dset["dev"], dset["test"] = (
        _ld("train.csv"),
        _ld("dev.csv"),
        _ld("test.csv"),
    )
    return dset


def make_synthetic(n_tr=300, n_dev=100, n_te=120):
    shapes = list(string.ascii_uppercase[:8])
    colors = list(string.ascii_lowercase[:8])

    def gen(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(4, 15)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seq = " ".join(toks)
            # simple parity rule: label 1 if majority shape==first shape
            maj = sum(t[0] == toks[0][0] for t in toks) > L / 2
            labels.append(int(maj))
            seqs.append(seq)
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_dict(gen(n_tr)),
            "dev": Dataset.from_dict(gen(n_dev)),
            "test": Dataset.from_dict(gen(n_te)),
        }
    )


dataset = try_load_benchmark() or make_synthetic()
print("Dataset sizes:", {k: len(v) for k, v in dataset.items()})

# ---------------------- vocabularies --------------------------
shape2idx, color2idx = {}, {}


def add_vocab(tok):
    sh, co = tok[0], tok[1]
    if sh not in shape2idx:
        shape2idx[sh] = len(shape2idx)
    if co not in color2idx:
        color2idx[co] = len(color2idx)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_vocab(tok)
n_shapes, n_colors = len(shape2idx), len(color2idx)
num_classes = len(set(dataset["train"]["label"]))
print(f"ShapeVocab={n_shapes}, ColorVocab={n_colors}, Classes={num_classes}")


# ------------------- seq -> graph -----------------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    shape_ids = [shape2idx[t[0]] for t in toks]
    color_ids = [color2idx[t[1]] for t in toks]
    pos_ids = list(range(len(toks)))
    # edges: consecutive
    src = list(range(len(toks) - 1))
    dst = list(range(1, len(toks)))
    # same-shape
    sh_dict = {}
    for i, sh in enumerate(shape_ids):
        sh_dict.setdefault(sh, []).append(i)
    for ids in sh_dict.values():
        for i in ids:
            for j in ids:
                if i < j:
                    src.append(i)
                    dst.append(j)
    # same-color
    co_dict = {}
    for i, co in enumerate(color_ids):
        co_dict.setdefault(co, []).append(i)
    for ids in co_dict.values():
        for i in ids:
            for j in ids:
                if i < j:
                    src.append(i)
                    dst.append(j)
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    x = torch.tensor(list(zip(shape_ids, color_ids)), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    data.seq_raw = seq
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


# ---------------------- model ---------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, n_shapes, n_colors, emb_dim=32, hid=64, n_cls=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb_dim)
        self.color_emb = nn.Embedding(n_colors, emb_dim)
        self.lin_in = nn.Linear(emb_dim * 2, hid)
        self.conv1 = SAGEConv(hid, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.out = nn.Linear(hid, n_cls)

    def forward(self, data):
        sh = self.shape_emb(data.x[:, 0])
        co = self.color_emb(data.x[:, 1])
        x = torch.cat([sh, co], dim=-1)
        x = F.relu(self.lin_in(x))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.out(x)


# ----------------- experiment logging -------------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------- training loop ---------------------------
def run_one(lr):
    model = GNNClassifier(n_shapes, n_colors, n_cls=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=64)
    for epoch in range(1, 9):
        # train
        model.train()
        tot_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
        tr_loss = tot_loss / len(train_loader.dataset)
        # eval
        model.eval()
        dev_loss = 0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = crit(logits, batch.y)
                dev_loss += loss.item() * batch.num_graphs
                p = logits.argmax(-1).cpu().tolist()
                g = batch.y.cpu().tolist()
                s = batch.seq_raw
                preds.extend(p)
                gts.extend(g)
                seqs.extend(s)
        dev_loss /= len(dev_loader.dataset)
        acc = np.mean([p == g for p, g in zip(preds, gts)])
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hpa = harmonic_poly_accuracy(cwa, swa)
        print(
            f"Epoch {epoch}: validation_loss = {dev_loss:.4f} | acc={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}"
        )
        # log
        experiment_data["SPR"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR"]["losses"]["val"].append(dev_loss)
        experiment_data["SPR"]["metrics"]["val"].append(
            {"acc": acc, "CWA": cwa, "SWA": swa, "HPA": hpa}
        )
        experiment_data["SPR"]["epochs"].append(epoch)
    experiment_data["SPR"]["predictions"] = preds
    experiment_data["SPR"]["ground_truth"] = gts


for lr in [1e-3, 3e-3]:
    print(f"\n=== Running LR={lr} ===")
    run_one(lr)

# ---------------------- save logs -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data ->", os.path.join(working_dir, "experiment_data.npy"))
