import os, time, copy, pathlib, numpy as np, torch, torch.nn as nn
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

# --------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- dataset helpers -----------------------------------
def locate_spr_bench() -> pathlib.Path:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]:
        if p and (p / "train.csv").exists():
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH not found")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


# ---------- metrics --------------------------------------------------
def _unique(seq, f):
    return len(set(f(tok) for tok in seq.split() if tok))


def count_color_variety(seq):
    return _unique(seq, lambda t: t[1] if len(t) > 1 else "")


def count_shape_variety(seq):
    return _unique(seq, lambda t: t[0])


def count_struct_complexity(seq):
    return len(set(seq.split()))


def _weighted_acc(weights, y_t, y_p):
    return sum(w if a == b else 0 for w, a, b in zip(weights, y_t, y_p)) / max(
        sum(weights), 1
    )


def cwa(seqs, y_t, y_p):
    return _weighted_acc([count_color_variety(s) for s in seqs], y_t, y_p)


def swa(seqs, y_t, y_p):
    return _weighted_acc([count_shape_variety(s) for s in seqs], y_t, y_p)


def strwa(seqs, y_t, y_p):
    return _weighted_acc([count_struct_complexity(s) for s in seqs], y_t, y_p)


# ---------- graph construction --------------------------------------
def build_vocab(dataset):
    shapes, colours, labels = set(), set(), set()
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok:
                shapes.add(tok[0])
                if len(tok) > 1:
                    colours.add(tok[1])
        labels.add(ex["label"])
    sh2i = {s: i for i, s in enumerate(sorted(shapes))}
    co2i = {c: i for i, c in enumerate(sorted(colours))}
    la2i = {l: i for i, l in enumerate(sorted(labels))}
    return sh2i, co2i, la2i


def seq_to_graph(ex, sh2i, co2i, la2i, max_len=60):
    toks = ex["sequence"].split()
    n = len(toks)
    shape_ids = [sh2i[t[0]] for t in toks]
    colour_ids = [co2i[t[1]] if len(t) > 1 else 0 for t in toks]
    pos_ids = list(range(n))
    x = torch.tensor(list(zip(shape_ids, colour_ids, pos_ids)), dtype=torch.long)

    src, dst, etype = [], [], []
    # relation 0: sequential neighbours
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [0, 0]
    # relation 1: same colour
    colour_map = {}
    for i, c in enumerate(colour_ids):
        colour_map.setdefault(c, []).append(i)
    for idxs in colour_map.values():
        for i in idxs:
            for j in idxs:
                if i < j:
                    src += [i, j]
                    dst += [j, i]
                    etype += [1, 1]
    # relation 2: same shape
    shape_map = {}
    for i, s in enumerate(shape_ids):
        shape_map.setdefault(s, []).append(i)
    for idxs in shape_map.values():
        for i in idxs:
            for j in idxs:
                if i < j:
                    src += [i, j]
                    dst += [j, i]
                    etype += [2, 2]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    y = torch.tensor([la2i[ex["label"]]], dtype=torch.long)
    g = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    g.seq = ex["sequence"]
    return g


# ----------------- load data ----------------------------------------
DATA = load_spr(locate_spr_bench())
shape2idx, colour2idx, label2idx = build_vocab(DATA["train"])
idx2label = {v: k for k, v in label2idx.items()}
MAX_POS = max(len(ex["sequence"].split()) for ex in DATA["train"])

train_g = [
    seq_to_graph(e, shape2idx, colour2idx, label2idx, MAX_POS) for e in DATA["train"]
]
dev_g = [
    seq_to_graph(e, shape2idx, colour2idx, label2idx, MAX_POS) for e in DATA["dev"]
]
test_g = [
    seq_to_graph(e, shape2idx, colour2idx, label2idx, MAX_POS) for e in DATA["test"]
]

train_loader = DataLoader(train_g, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_g, batch_size=128, shuffle=False)
test_loader = DataLoader(test_g, batch_size=128, shuffle=False)


# -------------- model ------------------------------------------------
class GraphTransformer(nn.Module):
    def __init__(
        self, n_shapes, n_colours, n_pos, n_labels, hid=64, heads=4, relations=3
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, hid)
        self.col_emb = nn.Embedding(max(n_colours, 1), hid)
        self.pos_emb = nn.Embedding(n_pos + 1, hid)
        self.rel_emb = nn.Embedding(relations, hid)
        self.conv1 = TransformerConv(hid, hid, heads=heads, edge_dim=hid, dropout=0.1)
        self.conv2 = TransformerConv(
            hid * heads, hid, heads=heads, edge_dim=hid, dropout=0.1
        )
        self.lin = nn.Linear(hid * heads, n_labels)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_type, batch):
        s, c, p = x.unbind(-1)
        h = self.shape_emb(s) + self.col_emb(c) + self.pos_emb(p)
        e = self.rel_emb(edge_type)
        h = self.conv1(h, edge_index, e).relu()
        h = self.conv2(h, edge_index, e).relu()
        h = global_mean_pool(h, batch)
        h = self.drop(h)
        return self.lin(h)


# ----------------- training utils -----------------------------------
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, preds, gold, seqs = 0.0, [], [], []
    for bt in loader:
        bt = bt.to(device)
        out = model(bt.x, bt.edge_index, bt.edge_type, bt.batch)
        loss = criterion(out, bt.y.view(-1))
        tot_loss += loss.item() * bt.num_graphs
        pr = out.argmax(-1).cpu().tolist()
        gt = bt.y.view(-1).cpu().tolist()
        preds.extend(pr)
        gold.extend(gt)
        seqs.extend(bt.seq)
    c = cwa(seqs, gold, preds)
    s = swa(seqs, gold, preds)
    r = strwa(seqs, gold, preds)
    bwa = (c + s) / 2.0
    return tot_loss / len(loader.dataset), bwa, c, s, r, preds, gold, seqs


# ------------------ experiment container ----------------------------
experiment_data = {
    "spr_bench": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------ training loop -----------------------------------
model = GraphTransformer(len(shape2idx), len(colour2idx), MAX_POS, len(label2idx)).to(
    device
)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
best_bwa, best_state, patience, wait = -1, None, 4, 0
MAX_EPOCH = 25

for epoch in range(1, MAX_EPOCH + 1):
    model.train()
    running = 0.0
    for bt in train_loader:
        bt = bt.to(device)
        opt.zero_grad()
        out = model(bt.x, bt.edge_index, bt.edge_type, bt.batch)
        loss = criterion(out, bt.y.view(-1))
        loss.backward()
        opt.step()
        running += loss.item() * bt.num_graphs
    train_loss = running / len(train_loader.dataset)
    val_loss, val_bwa, val_c, val_s, val_r, *_ = evaluate(model, dev_loader)
    tr_loss_eval, tr_bwa, tr_c, tr_s, tr_r, *_ = evaluate(model, train_loader)

    experiment_data["spr_bench"]["losses"]["train"].append(train_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)
    experiment_data["spr_bench"]["metrics"]["train"].append(
        {"BWA": tr_bwa, "CWA": tr_c, "SWA": tr_s, "StrWA": tr_r}
    )
    experiment_data["spr_bench"]["metrics"]["val"].append(
        {"BWA": val_bwa, "CWA": val_c, "SWA": val_s, "StrWA": val_r}
    )
    experiment_data["spr_bench"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: validation_loss = {val_loss:.4f}  "
        f"BWA={val_bwa:.3f} CWA={val_c:.3f} SWA={val_s:.3f} StrWA={val_r:.3f}"
    )

    if val_bwa > best_bwa:
        best_bwa, best_state, wait = val_bwa, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# -------------------- test evaluation -------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_bwa, test_c, test_s, test_r, preds, gold, seqs = evaluate(
    model, test_loader
)
experiment_data["spr_bench"]["predictions"] = preds
experiment_data["spr_bench"]["ground_truth"] = gold
experiment_data["spr_bench"]["test_metrics"] = {
    "loss": test_loss,
    "BWA": test_bwa,
    "CWA": test_c,
    "SWA": test_s,
    "StrWA": test_r,
}
print(
    f"TEST -> loss {test_loss:.4f}  BWA {test_bwa:.3f}  "
    f"CWA {test_c:.3f}  SWA {test_s:.3f}  StrWA {test_r:.3f}"
)

# -------------------- save ------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Experiment data saved.")
