import os, pathlib, random, itertools, time
import numpy as np
import torch, torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from collections import defaultdict
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------- misc setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- data utilities ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


# ---------- graph construction ----------
UNK = "<UNK>"


def build_vocab(seqs):
    toks = {tok for s in seqs for tok in s.split()}
    return {t: i + 1 for i, t in enumerate(sorted(toks))} | {UNK: 0}


def prepare_graphs(spr):
    shapes = sorted(
        {
            tok[0]
            for tok in itertools.chain.from_iterable(
                map(str.split, spr["train"]["sequence"])
            )
        }
    )
    colors = sorted(
        {
            tok[1]
            for tok in itertools.chain.from_iterable(
                map(str.split, spr["train"]["sequence"])
            )
        }
    )
    shape2i = {s: i for i, s in enumerate(shapes)}
    color2i = {c: i for i, c in enumerate(colors)}

    labels = sorted(set(spr["train"]["label"]))
    lab2i = {l: i for i, l in enumerate(labels)}

    def seq_to_graph(seq, label):
        tokens = seq.split()
        n = len(tokens)
        shape_idx = torch.tensor([shape2i[t[0]] for t in tokens], dtype=torch.long)
        color_idx = torch.tensor([color2i[t[1]] for t in tokens], dtype=torch.long)
        pos_idx = torch.tensor(list(range(n)), dtype=torch.long)
        # edges
        edge_set = set()
        for i in range(n - 1):  # adjacent
            edge_set.add((i, i + 1))
            edge_set.add((i + 1, i))
        groups = defaultdict(list)
        for i, t in enumerate(tokens):
            groups[("s", t[0])].append(i)
            groups[("c", t[1])].append(i)
        for g in groups.values():
            for i in g:
                for j in g:
                    if i != j:
                        edge_set.add((i, j))
        if not edge_set:
            edge_set.add((0, 0))
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
        x = torch.stack([shape_idx, color_idx, pos_idx], dim=1)
        return Data(x=x, edge_index=edge_index, y=torch.tensor([lab2i[label]]))

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = [
            seq_to_graph(s, l)
            for s, l in zip(spr[split]["sequence"], spr[split]["label"])
        ]
    return out, shape2i, color2i, lab2i


# ---------- model ----------
class GNNClassifier(nn.Module):
    def __init__(self, n_shape, n_color, max_pos, emb_dim, hid_dim, n_class):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(max_pos, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.lin = nn.Linear(hid_dim, n_class)

    def forward(self, data):
        s, c, p = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        x = self.shape_emb(s) + self.color_emb(c) + self.pos_emb(p)
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ---------- load dataset ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # fallback tiny synthetic

    def synth(n):
        shapes = "AB"
        colors = "12"
        seqs = [
            " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 8))
            )
            for _ in range(n)
        ]
        labels = [random.choice(["yes", "no"]) for _ in range(n)]
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    from datasets import Dataset

    spr = DatasetDict(
        {
            "train": Dataset.from_dict(synth(1000)),
            "dev": Dataset.from_dict(synth(200)),
            "test": Dataset.from_dict(synth(200)),
        }
    )
graphs, shape2i, color2i, lab2i = prepare_graphs(spr)
max_pos = max(len(g.x) for g in graphs["train"]) + 1
num_class = len(lab2i)
inv_lab = {v: k for k, v in lab2i.items()}

# ---------- loaders ----------
train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
dev_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)

# ---------- training ----------
model = GNNClassifier(
    len(shape2i), len(color2i), max_pos, emb_dim=32, hid_dim=64, n_class=num_class
).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
EPOCHS = 20


def evaluate(loader, seqs):
    model.eval()
    all_p, all_t = [], []
    loss_sum = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            loss_sum += loss.item() * batch.num_graphs
            preds = out.argmax(dim=-1).cpu().tolist()
            trues = batch.y.cpu().tolist()
            all_p.extend(preds)
            all_t.extend(trues)
    loss_avg = loss_sum / len(loader.dataset)
    pred_lbl = [inv_lab[p] for p in all_p]
    true_lbl = [inv_lab[t] for t in all_t]
    cwa = color_weighted_accuracy(seqs, true_lbl, pred_lbl)
    swa = shape_weighted_accuracy(seqs, true_lbl, pred_lbl)
    cpx = complexity_weighted_accuracy(seqs, true_lbl, pred_lbl)
    return loss_avg, cwa, swa, cpx, pred_lbl, true_lbl


for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    train_loss = epoch_loss / len(train_loader.dataset)
    # evaluate on dev
    val_loss, cwa, swa, cpx, _, _ = evaluate(dev_loader, spr["dev"]["sequence"])
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CpxWA={cpx:.3f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cpxwa": cpx}
    )

# ---------- final test ----------
test_loss, cwa_t, swa_t, cpx_t, preds_lbl, tru_lbl = evaluate(
    test_loader, spr["test"]["sequence"]
)
print(f"TEST  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}")
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cpxwa": cpx_t,
}
experiment_data["SPR_BENCH"]["predictions"] = preds_lbl
experiment_data["SPR_BENCH"]["ground_truth"] = tru_lbl

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
