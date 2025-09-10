import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ─────────── working dir & device ──────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────── metric helpers ────────────────────────────────────
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-6, sum(w))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-6, sum(w))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(1e-6, sum(w))


# ─────────── dataset loader (+ synthetic fallback) ─────────────
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ("train", "dev", "test"):
        d[split] = _load(f"{split}.csv")
    return d


def generate_synth(n: int):
    shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seqs.append(
            " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        )
        labels.append(random.randint(0, 2))
    return seqs, labels


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    spr = load_spr_bench(data_root)
    print("Loaded SPR_BENCH dataset.")
except Exception as e:
    print("Could not load real dataset, falling back to synthetic.", e)
    tr_s, tr_y = generate_synth(1000)
    dv_s, dv_y = generate_synth(200)
    ts_s, ts_y = generate_synth(200)
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


# ─────────── vocab build ───────────────────────────────────────
def build_vocabs(ds):
    shapes, colors, labels = set(), set(), set()
    for ex in ds:
        for tok in ex["sequence"].split():
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
        {l: i for i, l in enumerate(sorted(labels))},
    )


shape2i, color2i, label2i = build_vocabs(spr["train"])
NUM_SHAPE, NUM_COLOR, NUM_CLASS = len(shape2i), len(color2i), len(label2i)
MAX_POS = 20  # assume sequences shorter than this


# ─────────── graph conversion ─────────────────────────────────
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shape_idx = [shape2i[t[0]] for t in toks]
    color_idx = [color2i[t[1:]] for t in toks]
    pos_idx = list(range(n))
    # edges: consecutive + same color + same shape (undirected)
    src, dst = [], []
    # sequential
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    # same color/shape
    by_color, by_shape = {}, {}
    for i, t in enumerate(toks):
        by_color.setdefault(t[1:], []).append(i)
        by_shape.setdefault(t[0], []).append(i)

    def add_clique(idx_list):
        for i in idx_list:
            for j in idx_list:
                if i != j:
                    src.append(i)
                    dst.append(j)

    for lst in by_color.values():
        add_clique(lst)
    for lst in by_shape.values():
        add_clique(lst)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(list(zip(shape_idx, color_idx, pos_idx)), dtype=torch.long)
    y = torch.tensor([label2i[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["test"]]


# ─────────── model definition ────────────────────────────────
class SPRGraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 16
        hidden = 64
        self.shape_emb = nn.Embedding(NUM_SHAPE, emb_dim)
        self.color_emb = nn.Embedding(NUM_COLOR, emb_dim)
        self.pos_emb = nn.Embedding(MAX_POS, emb_dim)
        in_dim = emb_dim * 3
        self.g1 = SAGEConv(in_dim, hidden)
        self.g2 = SAGEConv(hidden, hidden)
        self.cls = nn.Linear(hidden, NUM_CLASS)

    def forward(self, data):
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        pos = self.pos_emb(data.x[:, 2].clamp(max=MAX_POS - 1))
        h = torch.cat([shp, col, pos], dim=-1)
        h = self.g1(h, data.edge_index).relu()
        h = self.g2(h, data.edge_index).relu()
        hg = global_mean_pool(h, data.batch)
        return self.cls(hg)


# ─────────── training utilities ──────────────────────────────
def run_training(epochs=20, batch_size=64):
    experiment_data = {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = SPRGraphNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_graphs, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * batch.num_graphs
        t_loss /= len(train_loader.dataset)
        # ---- val ----
        model.eval()
        v_loss = 0
        ys = []
        preds = []
        seqs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                v_loss += criterion(out, batch.y).item() * batch.num_graphs
                p = out.argmax(dim=1).cpu().tolist()
                preds.extend(p)
                ys.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq)
        v_loss /= len(val_loader.dataset)
        cwa = color_weighted_accuracy(seqs, ys, preds)
        swa = shape_weighted_accuracy(seqs, ys, preds)
        comp = complexity_weighted_accuracy(seqs, ys, preds)
        # logging
        print(
            f"Epoch {epoch}: val_loss={v_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )
        experiment_data["SPR"]["losses"]["train"].append(t_loss)
        experiment_data["SPR"]["losses"]["val"].append(v_loss)
        experiment_data["SPR"]["metrics"]["val"].append(
            {"CWA": cwa, "SWA": swa, "CompWA": comp}
        )
    # ---- final test evaluation ----
    test_loader = DataLoader(test_graphs, batch_size=128)
    model.eval()
    ys = []
    preds = []
    seqs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            o = model(batch)
            preds.extend(o.argmax(1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
            seqs.extend(batch.seq)
    experiment_data["SPR"]["predictions"] = preds
    experiment_data["SPR"]["ground_truth"] = ys
    experiment_data["SPR"]["metrics"]["test"] = {
        "CWA": color_weighted_accuracy(seqs, ys, preds),
        "SWA": shape_weighted_accuracy(seqs, ys, preds),
        "CompWA": complexity_weighted_accuracy(seqs, ys, preds),
    }
    return experiment_data


start = time.time()
exp_data = run_training()
print("Finished in", round(time.time() - start, 2), "s")
np.save(os.path.join(working_dir, "experiment_data.npy"), exp_data, allow_pickle=True)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
