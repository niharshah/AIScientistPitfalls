import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ────────────────── IO & DEVICE SET-UP ─────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# container for every run
experiment_data = {
    "baseline": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
    "uniform_node_feature": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
}


# ────────────────── METRICS ────────────────────────────────────
def count_color_variety(seq):
    return len(set(tok[1:] for tok in seq.split()))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def color_weighted_accuracy(seqs, y, g):
    wts = [count_color_variety(s) for s in seqs]
    return sum(w if yt == yp else 0 for w, yt, yp in zip(wts, y, g)) / max(
        1e-6, sum(wts)
    )


def shape_weighted_accuracy(seqs, y, g):
    wts = [count_shape_variety(s) for s in seqs]
    return sum(w if yt == yp else 0 for w, yt, yp in zip(wts, y, g)) / max(
        1e-6, sum(wts)
    )


def complexity_weighted_accuracy(seqs, y, g):
    wts = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(w if yt == yp else 0 for w, yt, yp in zip(wts, y, g)) / max(
        1e-6, sum(wts)
    )


# ────────────────── DATA LOADING ───────────────────────────────
def load_spr(root: pathlib.Path) -> DatasetDict:
    _f = lambda csv: load_dataset(
        "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict({s: _f(f"{s}.csv") for s in ["train", "dev", "test"]})


def make_synth(n):
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
    spr = load_spr(data_root)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic:", e)
    tr_s, tr_y = make_synth(1200)
    dv_s, dv_y = make_synth(300)
    ts_s, ts_y = make_synth(300)
    blank = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    spr = DatasetDict(
        {
            "train": blank.add_column("sequence", tr_s).add_column("label", tr_y),
            "dev": blank.add_column("sequence", dv_s).add_column("label", dv_y),
            "test": blank.add_column("sequence", ts_s).add_column("label", ts_y),
        }
    )


# ────────────────── VOCABS ─────────────────────────────────────
def build_voc(ds):
    shp, clr, lab = set(), set(), set()
    for ex in ds:
        for tok in ex["sequence"].split():
            shp.add(tok[0])
            clr.add(tok[1:])
        lab.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shp))},
        {c: i for i, c in enumerate(sorted(clr))},
        {l: i for i, l in enumerate(sorted(lab))},
    )


shape2i, color2i, label2i = build_voc(spr["train"])
NUM_SH, NUM_CL, NUM_LB = len(shape2i), len(color2i), len(label2i)
MAX_POS = 25


# ────────────────── GRAPH CONVERSION ──────────────────────────
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shp = [shape2i[t[0]] for t in toks]
    clr = [color2i[t[1:]] for t in toks]
    pos = list(range(n))
    src, dst, etype = [], [], []
    # sequential edges
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etype.extend([0, 0])
    # colour edges
    col_dict = {}
    for i, t in enumerate(toks):
        col_dict.setdefault(t[1:], []).append(i)
    for idxs in col_dict.values():
        for i in idxs:
            for j in idxs:
                if i < j:
                    src.extend([i, j])
                    dst.extend([j, i])
                    etype.extend([1, 1])
    # shape edges
    shp_dict = {}
    for i, t in enumerate(toks):
        shp_dict.setdefault(t[0], []).append(i)
    for idxs in shp_dict.values():
        for i in idxs:
            for j in idxs:
                if i < j:
                    src.extend([i, j])
                    dst.extend([j, i])
                    etype.extend([2, 2])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    x = torch.tensor(list(zip(shp, clr, pos)), dtype=torch.long)
    y = torch.tensor([label2i[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr["test"]]


# ────────────────── MODELS ─────────────────────────────────────
class SPR_RGCN(nn.Module):  # baseline
    def __init__(self, emb_dim=32, hid=128):
        super().__init__()
        self.shape_emb = nn.Embedding(NUM_SH, emb_dim)
        self.color_emb = nn.Embedding(NUM_CL, emb_dim)
        self.pos_emb = nn.Embedding(MAX_POS, emb_dim)
        self.rg1 = RGCNConv(emb_dim * 3, hid, num_relations=3)
        self.rg2 = RGCNConv(hid, hid, num_relations=3)
        self.cls = nn.Linear(hid, NUM_LB)

    def forward(self, bat):
        shp = self.shape_emb(bat.x[:, 0])
        col = self.color_emb(bat.x[:, 1])
        pos = self.pos_emb(bat.x[:, 2].clamp(max=MAX_POS - 1))
        h = torch.cat([shp, col, pos], dim=-1)
        h = self.rg1(h, bat.edge_index, bat.edge_type).relu()
        h = self.rg2(h, bat.edge_index, bat.edge_type).relu()
        return self.cls(global_mean_pool(h, bat.batch))


class SPR_RGCN_UNF(nn.Module):  # Uniform-Node-Feature ablation
    def __init__(self, emb_dim=32, hid=128):
        super().__init__()
        self.uni_emb = nn.Embedding(1, emb_dim * 2)  # shared learnable vector
        self.pos_emb = nn.Embedding(MAX_POS, emb_dim)  # keep positional signal
        self.rg1 = RGCNConv(emb_dim * 3, hid, num_relations=3)
        self.rg2 = RGCNConv(hid, hid, num_relations=3)
        self.cls = nn.Linear(hid, NUM_LB)

    def forward(self, bat):
        # repeat the same node feature for every token
        shared = self.uni_emb.weight.view(1, -1).repeat(bat.num_nodes, 1)
        pos = self.pos_emb(bat.x[:, 2].clamp(max=MAX_POS - 1))
        h = torch.cat([shared, pos], dim=-1)
        h = self.rg1(h, bat.edge_index, bat.edge_type).relu()
        h = self.rg2(h, bat.edge_index, bat.edge_type).relu()
        return self.cls(global_mean_pool(h, bat.batch))


# ────────────────── TRAIN / EVAL LOOP ──────────────────────────
def run_experiment(tag, model_cls, epochs=20, batch_size=64, lr=1e-3, patience=5):
    store = experiment_data[tag]["SPR"]
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_graphs, batch_size=batch_size)
    best, bad = 1e9, 0
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tl = 0
        for bat in tr_loader:
            bat = bat.to(device)
            opt.zero_grad()
            loss = crit(model(bat), bat.y)
            loss.backward()
            opt.step()
            tl += loss.item() * bat.num_graphs
        tl /= len(tr_loader.dataset)
        # ---- val ----
        model.eval()
        vl = 0
        ys = []
        ps = []
        seqs = []
        with torch.no_grad():
            for bat in val_loader:
                bat = bat.to(device)
                out = model(bat)
                vl += crit(out, bat.y).item() * bat.num_graphs
                ps.extend(out.argmax(1).cpu().tolist())
                ys.extend(bat.y.cpu().tolist())
                seqs.extend(bat.seq)
        vl /= len(val_loader.dataset)
        cwa = color_weighted_accuracy(seqs, ys, ps)
        swa = shape_weighted_accuracy(seqs, ys, ps)
        comp = complexity_weighted_accuracy(seqs, ys, ps)
        print(
            f"[{tag}] Epoch {ep}: val_loss {vl:.4f} | CWA {cwa:.3f} SWA {swa:.3f} CompWA {comp:.3f}"
        )
        store["losses"]["train"].append(tl)
        store["losses"]["val"].append(vl)
        store["metrics"]["val"].append({"CWA": cwa, "SWA": swa, "CompWA": comp})
        if vl < best:
            best = vl
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] Early stopping.")
                break
    # ---- test ----
    tst_loader = DataLoader(test_graphs, batch_size=128)
    model.eval()
    ys = []
    ps = []
    seqs = []
    with torch.no_grad():
        for bat in tst_loader:
            bat = bat.to(device)
            out = model(bat)
            ps.extend(out.argmax(1).cpu().tolist())
            ys.extend(bat.y.cpu().tolist())
            seqs.extend(bat.seq)
    store["predictions"] = ps
    store["ground_truth"] = ys
    store["metrics"]["test"] = {
        "CWA": color_weighted_accuracy(seqs, ys, ps),
        "SWA": shape_weighted_accuracy(seqs, ys, ps),
        "CompWA": complexity_weighted_accuracy(seqs, ys, ps),
    }
    print(f"[{tag}] TEST → {store['metrics']['test']}")


start = time.time()
run_experiment("baseline", SPR_RGCN)
run_experiment("uniform_node_feature", SPR_RGCN_UNF)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All results saved to", os.path.join(working_dir, "experiment_data.npy"))
print("Elapsed", round(time.time() - start, 2), "s")
