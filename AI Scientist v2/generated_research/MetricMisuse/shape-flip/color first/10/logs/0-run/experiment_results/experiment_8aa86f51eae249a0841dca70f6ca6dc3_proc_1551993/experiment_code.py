import os, pathlib, random, time, copy
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ------------ mandatory working dir & bookkeeping ------------
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

# ------------ device ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------ metrics ------------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    s = sum(w) or 1
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / s


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    s = sum(w) or 1
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / s


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    s = sum(w) or 1
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / s


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ------------ load SPR_BENCH ------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _l(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _l("train.csv"), "dev": _l("dev.csv"), "test": _l("test.csv")}
    )


if DATA_PATH.exists():
    spr = load_spr(DATA_PATH)
else:  # tiny synthetic fallback

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labs = [], []
        for _ in range(n):
            L = random.randint(4, 8)
            seqs.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            labs.append(random.choice(["yes", "no"]))
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    from datasets import Dataset

    spr = DatasetDict(
        {
            "train": Dataset.from_dict(synth(400)),
            "dev": Dataset.from_dict(synth(100)),
            "test": Dataset.from_dict(synth(100)),
        }
    )

# ------------ vocab & graph construction ------------
UNK = "<UNK>"


def build_vocab(seqs):
    vocab = {tok for s in seqs for tok in s.split()}
    return {t: i + 1 for i, t in enumerate(sorted(vocab))} | {UNK: 0}


token2idx = build_vocab(spr["train"]["sequence"])
labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}


def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    node_ids = torch.tensor([token2idx.get(t, 0) for t in toks], dtype=torch.long)
    # linear chain edges
    src = torch.arange(0, n - 1, dtype=torch.long)
    dst = src + 1
    edges = [torch.stack([src, dst], 0), torch.stack([dst, src], 0)]
    # same-shape / same-color edges
    shape_map, color_map = {}, {}
    for i, t in enumerate(toks):
        shape_map.setdefault(t[0], []).append(i)
        color_map.setdefault(t[1], []).append(i)

    def make_pairs(idx_list):
        if len(idx_list) < 2:
            return []
        comb = []
        base = idx_list
        for a in base:
            for b in base:
                if a != b:
                    comb.append((a, b))
        return comb

    extra = []
    for lst in shape_map.values():
        extra += make_pairs(lst)
    for lst in color_map.values():
        extra += make_pairs(lst)
    if extra:
        extra = torch.tensor(extra, dtype=torch.long).T
        edges.append(extra)
    edge_index = torch.cat(edges, 1) if edges else torch.empty((2, 0), dtype=torch.long)
    data = Data(x=node_ids, edge_index=edge_index, y=torch.tensor([label2idx[label]]))
    return data


def make_graphs(split):
    return [
        seq_to_graph(s, l) for s, l in zip(spr[split]["sequence"], spr[split]["label"])
    ]


graphs = {sp: make_graphs(sp) for sp in ["train", "dev", "test"]}


# ------------ model ------------
class GNN(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.conv1 = SAGEConv(emb, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch):
        x = self.emb(batch.x)
        x = F.relu(self.conv1(x, batch.edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, batch.edge_index))
        x = global_mean_pool(x, batch.batch)
        return self.lin(x)


# ------------ loaders ------------
train_loader = DataLoader(graphs["train"], batch_size=64, shuffle=True)
dev_loader = DataLoader(graphs["dev"], batch_size=128)
test_loader = DataLoader(graphs["test"], batch_size=128)


# ------------ training loop ------------
def evaluate(model, loader, seqs):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0
    with torch.no_grad():
        for i, b in enumerate(loader):
            seq_batch = seqs[
                i * loader.batch_size : i * loader.batch_size + b.num_graphs
            ]
            b = b.to(device)
            out = model(b)
            loss = F.cross_entropy(out, b.y)
            total_loss += loss.item() * b.num_graphs
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_true.extend(b.y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    p_lbl = [labels[i] for i in all_preds]
    t_lbl = [labels[i] for i in all_true]
    cwa = color_weighted_accuracy(seq_batch if len(loader) == 1 else seqs, t_lbl, p_lbl)
    swa = shape_weighted_accuracy(seq_batch if len(loader) == 1 else seqs, t_lbl, p_lbl)
    cpx = complexity_weighted_accuracy(
        seq_batch if len(loader) == 1 else seqs, t_lbl, p_lbl
    )
    return avg_loss, cwa, swa, cpx, p_lbl, t_lbl


epochs = 15
model = GNN(len(token2idx), classes=len(labels)).to(device)
opt = Adam(model.parameters(), lr=1e-3)

best_state, best_hwa = None, -1
for ep in range(1, epochs + 1):
    model.train()
    tot = 0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        opt.step()
        tot += loss.item() * batch.num_graphs
    tr_loss = tot / len(train_loader.dataset)
    val_loss, cwa, swa, cpx, _, _ = evaluate(model, dev_loader, spr["dev"]["sequence"])
    hwa = harmonic_weighted_accuracy(cwa, swa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": ep, "cwa": cwa, "swa": swa, "cpx": cpx, "hwa": hwa}
    )
    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  CWA={cwa:.3f}  SWA={swa:.3f}  CpxWA={cpx:.3f}  HWA={hwa:.3f}"
    )
    if hwa > best_hwa:
        best_hwa, best_state = hwa, copy.deepcopy(model.state_dict())

# ------------ test evaluation ------------
model.load_state_dict(best_state)
test_loss, cwa_t, swa_t, cpx_t, preds, true = evaluate(
    model, test_loader, spr["test"]["sequence"]
)
hwa_t = harmonic_weighted_accuracy(cwa_t, swa_t)
print(f"\nTEST  CWA={cwa_t:.3f}  SWA={swa_t:.3f}  CpxWA={cpx_t:.3f}  HWA={hwa_t:.3f}")

experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_t,
    "swa": swa_t,
    "cpx": cpx_t,
    "hwa": hwa_t,
}
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
