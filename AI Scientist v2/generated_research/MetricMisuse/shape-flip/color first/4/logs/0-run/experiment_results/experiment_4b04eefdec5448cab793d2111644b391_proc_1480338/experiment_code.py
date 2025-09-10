import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ───── working directory and device ───────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ───── helper metrics ─────────────────────────────────────────
def count_color_var(seq):
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_var(seq):
    return len(set(tok[0] for tok in seq.split()))


def CWA(seqs, y_true, y_pred):
    w = [count_color_var(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-6, sum(w))


def SWA(seqs, y_true, y_pred):
    w = [count_shape_var(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-6, sum(w))


def CompWA(seqs, y_true, y_pred):
    w = [count_shape_var(s) * count_color_var(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-6, sum(w))


# ───── dataset loading (with synthetic fallback) ──────────────
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _l(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({spl: _l(spl) for spl in ("train", "dev", "test")})


def synth(n: int):
    shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
    seqs, lab = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seqs.append(
            " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        )
        lab.append(random.randint(0, 2))
    return seqs, lab


root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    ds = load_spr(root)
    print("Loaded real SPR_BENCH")
except Exception as e:
    print("Dataset missing, generating synthetic", e)
    tr_s, tr_y = synth(1500)
    dv_s, dv_y = synth(300)
    ts_s, ts_y = synth(300)
    empty = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    ds = DatasetDict(
        {
            "train": empty.add_column("sequence", tr_s).add_column("label", tr_y),
            "dev": empty.add_column("sequence", dv_s).add_column("label", dv_y),
            "test": empty.add_column("sequence", ts_s).add_column("label", ts_y),
        }
    )


# ───── vocabulary construction ────────────────────────────────
def build_vocab(dataset):
    shapes, colors, labels = set(), set(), set()
    for ex in dataset:
        for tok in ex["sequence"].split():
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
        {l: i for i, l in enumerate(sorted(labels))},
    )


shape2i, color2i, label2i = build_vocab(ds["train"])
NUM_SHAPE, NUM_COLOR, NUM_CLASS = len(shape2i), len(color2i), len(label2i)
MAX_POS = 25


# ───── graph conversion with relation types ───────────────────
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shp = [shape2i[t[0]] for t in toks]
    col = [color2i[t[1:]] for t in toks]
    pos = list(range(n))
    src, dst, etype = [], [], []
    # type 0: sequential
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etype.extend([0, 0])
    # cliques by shape (type1) and color(type2)
    by_shape, by_color = {}, {}
    for i, t in enumerate(toks):
        by_shape.setdefault(t[0], []).append(i)
        by_color.setdefault(t[1:], []).append(i)

    def add(lst, rel):
        for i in lst:
            for j in lst:
                if i != j:
                    src.append(i)
                    dst.append(j)
                    etype.append(rel)

    for lst in by_shape.values():
        add(lst, 1)
    for lst in by_color.values():
        add(lst, 2)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    x = torch.tensor(list(zip(shp, col, pos)), dtype=torch.long)
    wt = float(count_shape_var(seq) * count_color_var(seq))
    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor(label2i[label]),
        seq=seq,
        weight=torch.tensor(wt, dtype=torch.float),
    )


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in ds["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in ds["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in ds["test"]]


# ───── model definition ───────────────────────────────────────
class RelSPRNet(nn.Module):
    def __init__(self):
        super().__init__()
        emb = 24
        hid = 96
        self.emb_shape = nn.Embedding(NUM_SHAPE, emb)
        self.emb_color = nn.Embedding(NUM_COLOR, emb)
        self.emb_pos = nn.Embedding(MAX_POS, emb)
        in_dim = emb * 3
        self.rgcn1 = RGCNConv(in_dim, hid, num_relations=3)
        self.rgcn2 = RGCNConv(hid, hid, num_relations=3)
        self.lin = nn.Linear(hid, NUM_CLASS)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        h = torch.cat(
            [
                self.emb_shape(data.x[:, 0]),
                self.emb_color(data.x[:, 1]),
                self.emb_pos(data.x[:, 2].clamp(max=MAX_POS - 1)),
            ],
            dim=-1,
        )
        h = self.rgcn1(h, data.edge_index, data.edge_type).relu()
        h = self.dropout(h)
        h = self.rgcn2(h, data.edge_index, data.edge_type).relu()
        hg = global_mean_pool(h, data.batch)
        return self.lin(hg)


# ───── training loop ──────────────────────────────────────────
def train_model(epochs=10, batch_size=64):
    exp = {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = RelSPRNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction="none")
    tr_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_graphs, batch_size=batch_size)
    for epoch in range(1, epochs + 1):
        model.train()
        tloss = 0
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = (criterion(out, batch.y) * batch.weight.to(device)).mean()
            loss.backward()
            opt.step()
            tloss += loss.item() * batch.num_graphs
        tloss /= len(tr_loader.dataset)
        # validation
        model.eval()
        vloss = 0
        ys = []
        preds = []
        seqs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = (criterion(out, batch.y) * batch.weight.to(device)).mean()
                vloss += loss.item() * batch.num_graphs
                p = out.argmax(dim=1).cpu().tolist()
                preds.extend(p)
                ys.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq)
        vloss /= len(val_loader.dataset)
        cwa, swa, comp = (
            CWA(seqs, ys, preds),
            SWA(seqs, ys, preds),
            CompWA(seqs, ys, preds),
        )
        print(
            f"Epoch {epoch}: validation_loss = {vloss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )
        exp["SPR"]["losses"]["train"].append(tloss)
        exp["SPR"]["losses"]["val"].append(vloss)
        exp["SPR"]["metrics"]["val"].append({"CWA": cwa, "SWA": swa, "CompWA": comp})
    # test
    test_loader = DataLoader(test_graphs, batch_size=128)
    model.eval()
    ys = []
    preds = []
    seqs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds.extend(out.argmax(1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
            seqs.extend(batch.seq)
    exp["SPR"]["predictions"] = preds
    exp["SPR"]["ground_truth"] = ys
    exp["SPR"]["metrics"]["test"] = {
        "CWA": CWA(seqs, ys, preds),
        "SWA": SWA(seqs, ys, preds),
        "CompWA": CompWA(seqs, ys, preds),
    }
    print("Test metrics:", exp["SPR"]["metrics"]["test"])
    return exp


start = time.time()
experiment_data = train_model()
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics ->", os.path.join(working_dir, "experiment_data.npy"))
print("Finished in", round(time.time() - start, 2), "s")
