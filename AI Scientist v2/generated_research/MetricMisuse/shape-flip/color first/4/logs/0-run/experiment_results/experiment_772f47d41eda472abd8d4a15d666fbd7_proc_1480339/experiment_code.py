import os, pathlib, random, string, time, numpy as np, torch, torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ───────── working dir & device ───────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ───────── metrics ────────────────────────────────────────────
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-8, sum(w))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-8, sum(w))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-8, sum(w))


# ───────── dataset util ───────────────────────────────────────
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    dd = DatasetDict()
    for split in ("train", "dev", "test"):
        dd[split] = _load(f"{split}.csv")
    return dd


def generate_synthetic(n):
    shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seqs.append(
            " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        )
        labels.append(random.randint(0, 2))
    return seqs, labels


def build_vocabs(dataset_dict):
    shapes, colors, labels = set(), set(), set()
    for split in dataset_dict.values():
        for ex in split:
            for tok in ex["sequence"].split():
                shapes.add(tok[0])
                colors.add(tok[1:])
            labels.add(ex["label"])
    shape2i = {s: i for i, s in enumerate(sorted(shapes))}
    color2i = {c: i for i, c in enumerate(sorted(colors))}
    label2i = {l: i for i, l in enumerate(sorted(labels))}
    return shape2i, color2i, label2i


# ───────── load or create data ────────────────────────────────
root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    spr = load_spr_bench(root)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic dataset.", e)
    tr_s, tr_y = generate_synthetic(1000)
    dv_s, dv_y = generate_synthetic(200)
    ts_s, ts_y = generate_synthetic(200)
    empty = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    spr = DatasetDict(
        {
            "train": empty.add_column("sequence", tr_s).add_column("label", tr_y),
            "dev": empty.add_column("sequence", dv_s).add_column("label", dv_y),
            "test": empty.add_column("sequence", ts_s).add_column("label", ts_y),
        }
    )

shape2i, color2i, label2i = build_vocabs(spr)
NUM_SHAPE, NUM_COLOR, NUM_CLASS = len(shape2i), len(color2i), len(label2i)
MAX_POS = 25  # cap positional embedding


# ───────── graph conversion ───────────────────────────────────
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shape_idx = [shape2i[t[0]] for t in toks]
    color_idx = [color2i[t[1:]] for t in toks]
    pos_idx = list(range(n))
    src, dst, etype = [], [], []  # edge types: 0=seq,1=color,2=shape
    # sequential bi-directional
    for i in range(n - 1):
        for u, v in ((i, i + 1), (i + 1, i)):
            src.append(u)
            dst.append(v)
            etype.append(0)
    by_color, by_shape = {}, {}
    for i, t in enumerate(toks):
        by_color.setdefault(t[1:], []).append(i)
        by_shape.setdefault(t[0], []).append(i)

    def add_full(idx_list, tp):
        for i in idx_list:
            for j in idx_list:
                if i != j:
                    src.append(i)
                    dst.append(j)
                    etype.append(tp)

    for lst in by_color.values():
        add_full(lst, 1)
    for lst in by_shape.values():
        add_full(lst, 2)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    x = torch.tensor(list(zip(shape_idx, color_idx, pos_idx)), dtype=torch.long)
    y = torch.tensor([label2i[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["test"]]


# ───────── model ──────────────────────────────────────────────
class SPR_RGCN(nn.Module):
    def __init__(self, embed_dim=32, hidden=64, n_rel=3):
        super().__init__()
        self.shape_emb = nn.Embedding(NUM_SHAPE, embed_dim)
        self.color_emb = nn.Embedding(NUM_COLOR, embed_dim)
        self.pos_emb = nn.Embedding(MAX_POS, embed_dim)
        in_dim = embed_dim
        self.conv1 = RGCNConv(in_dim, hidden, num_relations=n_rel)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=n_rel)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.cls = nn.Linear(hidden, NUM_CLASS)

    def forward(self, data):
        # node feature
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        pos = self.pos_emb(torch.clamp(data.x[:, 2], max=MAX_POS - 1))
        x = shp + col + pos  # (N,embed_dim)
        x = F.relu(self.bn1(self.conv1(x, data.edge_index, data.edge_type)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, data.edge_index, data.edge_type)))
        hg = global_mean_pool(x, data.batch)  # graph-level
        return self.cls(hg)


# ───────── training loop ──────────────────────────────────────
def run(epochs=10, bs=64, lr=1e-3):
    experiment_data = {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = SPR_RGCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    vl = DataLoader(dev_graphs, batch_size=bs)
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t_loss = 0
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y)
            loss.backward()
            opt.step()
            t_loss += loss.item() * batch.num_graphs
        t_loss /= len(tl.dataset)
        # ---- validate ----
        model.eval()
        v_loss = 0
        ys = []
        ps = []
        seqs = []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                o = model(batch)
                v_loss += crit(o, batch.y).item() * batch.num_graphs
                ps.extend(o.argmax(1).cpu().tolist())
                ys.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq)
        v_loss /= len(vl.dataset)
        cwa = color_weighted_accuracy(seqs, ys, ps)
        swa = shape_weighted_accuracy(seqs, ys, ps)
        comp = complexity_weighted_accuracy(seqs, ys, ps)
        print(
            f"Epoch {epoch}: validation_loss = {v_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )
        # log
        experiment_data["SPR"]["losses"]["train"].append(t_loss)
        experiment_data["SPR"]["losses"]["val"].append(v_loss)
        experiment_data["SPR"]["metrics"]["val"].append(
            {"CWA": cwa, "SWA": swa, "CompWA": comp}
        )
    # ---- test ----
    model.eval()
    ys = []
    ps = []
    seqs = []
    tl_test = DataLoader(test_graphs, batch_size=128)
    with torch.no_grad():
        for batch in tl_test:
            batch = batch.to(device)
            o = model(batch)
            ps.extend(o.argmax(1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
            seqs.extend(batch.seq)
    experiment_data["SPR"]["predictions"] = ps
    experiment_data["SPR"]["ground_truth"] = ys
    experiment_data["SPR"]["metrics"]["test"] = {
        "CWA": color_weighted_accuracy(seqs, ys, ps),
        "SWA": shape_weighted_accuracy(seqs, ys, ps),
        "CompWA": complexity_weighted_accuracy(seqs, ys, ps),
    }
    return experiment_data


start = time.time()
exp_data = run()
print("Finished in", round(time.time() - start, 2), "s")
np.save(os.path.join(working_dir, "experiment_data.npy"), exp_data, allow_pickle=True)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
