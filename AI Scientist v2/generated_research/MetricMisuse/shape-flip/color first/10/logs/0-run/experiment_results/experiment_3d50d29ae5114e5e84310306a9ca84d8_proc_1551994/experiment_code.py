import os, pathlib, itertools, random, copy, time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from datasets import DatasetDict, load_dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# -------------------- house-keeping --------------------
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

# -------------------- device ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- metric helpers -------------------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


# -------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


# -------------------- graph builders -------------------
UNK = "<UNK>"


def build_vocab(sequences):
    vocab = {tok for seq in sequences for tok in seq.split()}
    token2idx = {t: i + 1 for i, t in enumerate(sorted(vocab))}
    token2idx[UNK] = 0
    return token2idx


def sequence_to_graph(seq, token2idx, label_idx):
    toks = seq.strip().split()
    n = len(toks)
    # base sequential edges
    edges = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    # same shape edges
    shape_groups = {}
    color_groups = {}
    for i, tok in enumerate(toks):
        shape_groups.setdefault(tok[0], []).append(i)
        color_groups.setdefault(tok[1], []).append(i)
    for group in list(shape_groups.values()) + list(color_groups.values()):
        for i, j in itertools.combinations(group, 2):
            edges.extend([(i, j), (j, i)])
    if not edges:  # singleton
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor([token2idx.get(t, token2idx[UNK]) for t in toks], dtype=torch.long)
    data = Data(
        x=x, edge_index=edge_index, y=torch.tensor([label_idx], dtype=torch.long)
    )
    data.seq = seq
    return data


def prepare_graph_sets(dset):
    token2idx = build_vocab(dset["train"]["sequence"])
    labels = sorted(set(dset["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(labels)}

    def _convert(split):
        return [
            sequence_to_graph(s, token2idx, label2idx[l])
            for s, l in zip(dset[split]["sequence"], dset[split]["label"])
        ]

    return (
        {split: _convert(split) for split in ["train", "dev", "test"]},
        token2idx,
        label2idx,
    )


# -------------------- model ----------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, nclass=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.conv1 = GATConv(emb, hid // heads, heads=heads)
        self.conv2 = GATConv(hid, hid // heads, heads=heads)
        self.lin = nn.Linear(hid, nclass)

    def forward(self, data):
        x = self.emb(data.x)
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# -------------------- data -----------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # fallback tiny synthetic set for demo/debug
    print("Real dataset not found, creating synthetic demo data.")

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seq, lbl = [], []
        for i in range(n):
            L = random.randint(4, 8)
            seq.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            lbl.append(random.choice(["yes", "no"]))
        return {"id": list(range(n)), "sequence": seq, "label": lbl}

    from datasets import Dataset

    spr = DatasetDict(
        {
            "train": Dataset.from_dict(synth(500)),
            "dev": Dataset.from_dict(synth(100)),
            "test": Dataset.from_dict(synth(100)),
        }
    )

graph_sets, token2idx, label2idx = prepare_graph_sets(spr)
inv_label = {v: k for k, v in label2idx.items()}
num_classes = len(label2idx)
print(f"Vocab size {len(token2idx)} | Classes {num_classes}")

train_loader = DataLoader(graph_sets["train"], batch_size=128, shuffle=True)
dev_loader = DataLoader(graph_sets["dev"], batch_size=256)
test_loader = DataLoader(graph_sets["test"], batch_size=256)


# -------------------- train / eval ----------------------
def evaluate(model, loader):
    model.eval()
    total_loss, seqs, preds, trues = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(-1).cpu().tolist()
            true = batch.y.cpu().tolist()
            preds.extend(pred)
            trues.extend(true)
            seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    pred_lbl = [inv_label[p] for p in preds]
    true_lbl = [inv_label[t] for t in trues]
    cwa = color_weighted_accuracy(seqs, true_lbl, pred_lbl)
    swa = shape_weighted_accuracy(seqs, true_lbl, pred_lbl)
    cpx = complexity_weighted_accuracy(seqs, true_lbl, pred_lbl)
    return avg_loss, cwa, swa, cpx, pred_lbl, true_lbl


# -------------------- training loop --------------------
EPOCHS = 25
model = GNNClassifier(len(token2idx), nclass=num_classes).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

best_cpx, best_state = -1.0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
    train_loss = tot_loss / len(train_loader.dataset)

    val_loss, cwa, swa, cpx, _, _ = evaluate(model, dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"cwa": cwa, "swa": swa, "cpx": cpx}
    )

    print(
        f"Epoch {epoch:02d}: validation_loss = {val_loss:.4f} | CWA {cwa:.3f} | SWA {swa:.3f} | CpxWA {cpx:.3f}"
    )

    if cpx > best_cpx:
        best_cpx = cpx
        best_state = copy.deepcopy(model.state_dict())

# -------------------- test evaluation ------------------
model.load_state_dict(best_state)
test_loss, cwa_t, swa_t, cpx_t, preds, true = evaluate(model, test_loader)
print(f"\nTEST  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}")

experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cpx": cpx_t,
}
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
