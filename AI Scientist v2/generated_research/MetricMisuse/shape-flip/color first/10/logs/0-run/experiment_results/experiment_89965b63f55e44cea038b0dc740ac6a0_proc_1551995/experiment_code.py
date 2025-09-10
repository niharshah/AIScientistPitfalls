import os, pathlib, itertools, random, time
import numpy as np
import torch, torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---------- working dir & device ----------
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


# ---------- dataset helpers ----------
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


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


# ---------- graph construction ----------
def prepare_graphs(spr):
    shapes = sorted({tok[0] for seq in spr["train"]["sequence"] for tok in seq.split()})
    colors = sorted({tok[1] for seq in spr["train"]["sequence"] for tok in seq.split()})
    shape2i, color2i = {s: i for i, s in enumerate(shapes)}, {
        c: i for i, c in enumerate(colors)
    }
    labels = sorted(set(spr["train"]["label"]))
    lab2i = {l: i for i, l in enumerate(labels)}

    def seq_to_graph(seq, label):
        tokens = seq.split()
        n = len(tokens)
        if n == 0:
            tokens = ["A1"]
            n = 1
        shape_idx = torch.tensor([shape2i[t[0]] for t in tokens], dtype=torch.long)
        color_idx = torch.tensor([color2i[t[1]] for t in tokens], dtype=torch.long)
        pos_idx = torch.tensor(list(range(n)), dtype=torch.long)

        edge_src, edge_dst, edge_type = [], [], []
        # type 0: adjacency
        for i in range(n - 1):
            edge_src += [i, i + 1]
            edge_dst += [i + 1, i]
            edge_type += [0, 0]
        # type 1: same shape
        buckets = {}
        for i, t in enumerate(tokens):
            buckets.setdefault(t[0], []).append(i)
        for idxs in buckets.values():
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_type.append(1)
        # type 2: same color
        buckets = {}
        for i, t in enumerate(tokens):
            buckets.setdefault(t[1], []).append(i)
        for idxs in buckets.values():
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_type.append(2)
        if not edge_src:  # safeguard single-token
            edge_src, edge_dst, edge_type = [0], [0], [0]
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        x = torch.stack([shape_idx, color_idx, pos_idx], dim=1)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([lab2i[label]], dtype=torch.long),
        )

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = [
            seq_to_graph(s, l)
            for s, l in zip(spr[split]["sequence"], spr[split]["label"])
        ]
    return out, shape2i, color2i, lab2i


# ---------- model ----------
class RelGNN(nn.Module):
    def __init__(self, n_shape, n_color, max_pos, emb_dim, hid_dim, n_classes):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(max_pos, emb_dim)
        self.conv1 = RGCNConv(emb_dim, hid_dim, num_relations=3)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_relations=3)
        self.lin = nn.Linear(hid_dim, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        s, c, p = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        x = self.shape_emb(s) + self.color_emb(c) + self.pos_emb(p)
        x = self.dropout(self.conv1(x, data.edge_index, data.edge_type).relu())
        x = self.dropout(self.conv2(x, data.edge_index, data.edge_type).relu())
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ---------- load data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr_raw = load_spr_bench(DATA_PATH)
else:  # small synthetic fallback for demo

    def synth(n):
        shapes, colors = "AB", "12"
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

    spr_raw = DatasetDict(
        {
            "train": Dataset.from_dict(synth(1000)),
            "dev": Dataset.from_dict(synth(200)),
            "test": Dataset.from_dict(synth(200)),
        }
    )
graphs, shape2i, color2i, lab2i = prepare_graphs(spr_raw)
inv_lab = {v: k for k, v in lab2i.items()}
max_pos = max(len(g.x) for g in graphs["train"]) + 1

train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
dev_loader = DataLoader(graphs["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)

# ---------- training setup ----------
model = RelGNN(
    len(shape2i), len(color2i), max_pos, emb_dim=32, hid_dim=64, n_classes=len(lab2i)
).to(device)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS = 25
best_val_loss, patience, wait = float("inf"), 5, 0


# ---------- evaluation helper ----------
def evaluate(loader, seqs):
    model.eval()
    loss_sum = 0
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            loss_sum += loss.item() * batch.num_graphs
            all_p.extend(out.argmax(dim=-1).cpu().tolist())
            all_t.extend(batch.y.cpu().tolist())
    avg_loss = loss_sum / len(loader.dataset)
    pred_lbl = [inv_lab[p] for p in all_p]
    true_lbl = [inv_lab[t] for t in all_t]
    cwa = color_weighted_accuracy(seqs, true_lbl, pred_lbl)
    swa = shape_weighted_accuracy(seqs, true_lbl, pred_lbl)
    cpx = complexity_weighted_accuracy(seqs, true_lbl, pred_lbl)
    return avg_loss, cwa, swa, cpx, pred_lbl, true_lbl


# ---------- training loop ----------
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
    val_loss, cwa, swa, cpx, _, _ = evaluate(dev_loader, spr_raw["dev"]["sequence"])
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CpxWA={cpx:.3f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cpxwa": cpx}
    )

    # early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss, wait = val_loss, 0
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping")
        break

# ---------- final evaluation on test ----------
test_loss, cwa_t, swa_t, cpx_t, preds_lbl, tru_lbl = evaluate(
    test_loader, spr_raw["test"]["sequence"]
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
