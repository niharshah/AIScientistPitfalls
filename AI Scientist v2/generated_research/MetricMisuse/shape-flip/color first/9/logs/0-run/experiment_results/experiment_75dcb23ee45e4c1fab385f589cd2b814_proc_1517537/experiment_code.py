import os, random, pathlib, itertools, time
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import RGCNConv, global_mean_pool

# ------------------------------------------------------------------#
# mandatory working dir and device                                  #
# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# container for all logged data
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------#
# Metric helpers                                                    #
# ------------------------------------------------------------------#
def _uniq_colors(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _uniq_shapes(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def CWA(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    w = [_uniq_colors(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def SWA(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    w = [_uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def HWA(cwa: float, swa: float, eps: float = 1e-12) -> float:
    return 2 * cwa * swa / (cwa + swa + eps)


# ------------------------------------------------------------------#
# Dataset loading (real or synthetic fallback)                      #
# ------------------------------------------------------------------#
def try_load_real():
    """
    Attempt to load the real SPR_BENCH dataset via SPR.py utility.
    Falls back to synthetic data if anything goes wrong.
    """
    try:
        import SPR  # assuming SPR.py is in PYTHONPATH / same folder

        path = pathlib.Path("./SPR_BENCH")
        dset = SPR.load_spr_bench(path)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        print("Real SPR_BENCH not found – using synthetic. Reason:", e)
        return None


def gen_synth(n: int = 1500) -> Dict[str, List[str]]:
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(n):
        ln = random.randint(4, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(toks)
        label = (_uniq_shapes(seq) + _uniq_colors(seq)) % 4  # toy rule
        seqs.append(seq)
        labels.append(label)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real
else:
    train_raw, dev_raw, test_raw = (
        gen_synth(3000),
        gen_synth(600),
        gen_synth(600),
    )


# ------------------------------------------------------------------#
# Build vocabularies                                                #
# ------------------------------------------------------------------#
def build_vocabs(dsets):
    shapes, colors = set(), set()
    for data in dsets:
        seq_iter = (
            data["sequence"]
            if isinstance(data, dict)
            else (ex["sequence"] for ex in data)
        )
        for s in seq_iter:
            for tok in s.split():
                shapes.add(tok[0])
                colors.add(tok[1])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
    )


shape_vocab, color_vocab = build_vocabs([train_raw, dev_raw, test_raw])
S, C = len(shape_vocab), len(color_vocab)


# ------------------------------------------------------------------#
# Utilities                                                         #
# ------------------------------------------------------------------#
def _add_bidirected(i: int, j: int, rel: int, e_idx: list, e_rel: list):
    e_idx.extend([(i, j), (j, i)])
    e_rel.extend([rel, rel])


# ------------------------------------------------------------------#
# Sequence -> PyG graph                                             #
# ------------------------------------------------------------------#
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    # Node features
    s_ids = torch.tensor([shape_vocab[t[0]] for t in toks], dtype=torch.long)
    c_ids = torch.tensor([color_vocab[t[1]] for t in toks], dtype=torch.long)
    pos = torch.arange(n, dtype=torch.float32) / (n - 1 if n > 1 else 1)

    x = torch.cat(
        [
            nn.functional.one_hot(s_ids, num_classes=S),
            nn.functional.one_hot(c_ids, num_classes=C),
            pos.unsqueeze(1),
        ],
        dim=1,
    ).float()

    edges, rels = [], []

    # 0: sequential
    for i in range(n - 1):
        _add_bidirected(i, i + 1, 0, edges, rels)

    # 1: same shape (use unordered pairs instead of permutations)
    for sh in set(s_ids.tolist()):
        idx = [i for i, v in enumerate(s_ids) if v == sh]
        for i, j in itertools.combinations(idx, 2):
            _add_bidirected(i, j, 1, edges, rels)

    # 2: same color
    for co in set(c_ids.tolist()):
        idx = [i for i, v in enumerate(c_ids) if v == co]
        for i, j in itertools.combinations(idx, 2):
            _add_bidirected(i, j, 2, edges, rels)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(rels, dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([int(label)], dtype=torch.long),
        seq=seq,
    )


def build_pyg(raw):
    if isinstance(raw, dict):
        return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]
    # HF datasets case
    return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in raw]


train_ds, dev_ds, test_ds = map(build_pyg, (train_raw, dev_raw, test_raw))
num_classes = len({d.y.item() for d in train_ds})


# ------------------------------------------------------------------#
# Model                                                             #
# ------------------------------------------------------------------#
class SPR_RGCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, num_rel: int, classes: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hid, num_rel)
        self.conv2 = RGCNConv(hid, hid, num_rel)
        self.lin = nn.Linear(hid, classes)

    def forward(self, data: Batch):
        x, ei, et, b = data.x, data.edge_index, data.edge_type, data.batch
        x = self.conv1(x, ei, et).relu()
        x = self.conv2(x, ei, et).relu()
        x = global_mean_pool(x, b)
        return self.lin(x)


# ------------------------------------------------------------------#
# Training utilities                                                #
# ------------------------------------------------------------------#
def run_epoch(model, loader, criterion, opt=None):
    training = opt is not None
    model.train() if training else model.eval()
    total_loss, seqs, ys, ps = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        if training:
            opt.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.squeeze())
        if training:
            loss.backward()
            opt.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(1).cpu().tolist()
        labels = batch.y.cpu().view(-1).tolist()
        seqs.extend(batch.seq)
        ys.extend(labels)
        ps.extend(preds)
    avg_loss = total_loss / len(loader.dataset)
    cwa, swa = CWA(seqs, ys, ps), SWA(seqs, ys, ps)
    hwa = HWA(cwa, swa)
    return avg_loss, {"CWA": cwa, "SWA": swa, "HWA": hwa}, ys, ps


# ------------------------------------------------------------------#
# Training loop                                                     #
# ------------------------------------------------------------------#
EPOCHS = 15
LR = 5e-4
BATCH_TRAIN, BATCH_EVAL = 32, 64
criterion = nn.CrossEntropyLoss()

model = SPR_RGCN(S + C + 1, 64, 3, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
val_loader = DataLoader(dev_ds, batch_size=BATCH_EVAL)

best_state, best_hwa = None, -1.0
for epoch in range(1, EPOCHS + 1):
    t_loss, t_met, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    v_loss, v_met, _, _ = run_epoch(model, val_loader, criterion)
    print(
        f"Epoch {epoch:02d}: val_loss = {v_loss:.4f} | "
        f"CWA = {v_met['CWA']:.4f} | SWA = {v_met['SWA']:.4f} | HWA = {v_met['HWA']:.4f}"
    )

    experiment_data["SPR"]["losses"]["train"].append(t_loss)
    experiment_data["SPR"]["losses"]["val"].append(v_loss)
    experiment_data["SPR"]["metrics"]["train"].append(t_met)
    experiment_data["SPR"]["metrics"]["val"].append(v_met)
    experiment_data["SPR"]["epochs"].append(epoch)

    if v_met["HWA"] > best_hwa:
        best_hwa = v_met["HWA"]
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ------------------------------------------------------------------#
# Test evaluation                                                   #
# ------------------------------------------------------------------#
model.load_state_dict(best_state)
test_loader = DataLoader(test_ds, batch_size=BATCH_EVAL)
_, test_met, gts, prs = run_epoch(model, test_loader, criterion)
print(
    f"Test  CWA = {test_met['CWA']:.4f} | SWA = {test_met['SWA']:.4f} | HWA = {test_met['HWA']:.4f}"
)

experiment_data["SPR"]["predictions"] = prs
experiment_data["SPR"]["ground_truth"] = gts

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
