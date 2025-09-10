import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ────────────────── WORK DIR & DEVICE ─────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────── EXPERIMENT DATA CONTAINER ─────────────────
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ────────────────── METRIC HELPERS ────────────────────────────
def count_color_variety(seq: str) -> int:
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        1e-9, sum(w)
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        1e-9, sum(w)
    )


def dual_weighted_accuracy(seqs, y_true, y_pred):
    return 0.5 * (
        color_weighted_accuracy(seqs, y_true, y_pred)
        + shape_weighted_accuracy(seqs, y_true, y_pred)
    )


# ────────────────── DATA LOADING ──────────────────────────────
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({k: _load(f"{k}.csv") for k in ["train", "dev", "test"]})


def make_synthetic(n: int):
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
    spr_dataset = load_spr_bench(data_root)
except Exception as e:
    print("Falling back to synthetic data:", e)
    tr_s, tr_y = make_synthetic(2000)
    dv_s, dv_y = make_synthetic(400)
    ts_s, ts_y = make_synthetic(400)
    blank = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    spr_dataset = DatasetDict(
        {
            "train": blank.add_column("sequence", tr_s).add_column("label", tr_y),
            "dev": blank.add_column("sequence", dv_s).add_column("label", dv_y),
            "test": blank.add_column("sequence", ts_s).add_column("label", ts_y),
        }
    )

print({k: len(v) for k, v in spr_dataset.items()})


# ────────────────── VOCAB BUILDING ────────────────────────────
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


shape2i, color2i, label2i = build_voc(spr_dataset["train"])
NUM_SH, NUM_CL, NUM_LB, MAX_POS = len(shape2i), len(color2i), len(label2i), 25


# ────────────────── GRAPH CONVERSION ─────────────────────────-
def seq_to_graph(seq, label):
    toks = seq.split()
    shp = [shape2i[t[0]] for t in toks]
    clr = [color2i[t[1:]] for t in toks]
    pos = list(range(len(toks)))

    # Relations: same color (1) and same shape (2)
    src, dst, etype = [], [], []
    color_groups, shape_groups = {}, {}
    for idx, tok in enumerate(toks):
        color_groups.setdefault(tok[1:], []).append(idx)
        shape_groups.setdefault(tok[0], []).append(idx)

    def add_full_connect(indices, rel_id):
        for i in indices:
            for j in indices:
                if i < j:
                    src.extend([i, j])
                    dst.extend([j, i])
                    etype.extend([rel_id, rel_id])

    for g in color_groups.values():
        add_full_connect(g, 1)
    for g in shape_groups.values():
        add_full_connect(g, 2)

    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = (
        torch.tensor(etype, dtype=torch.long)
        if etype
        else torch.empty((0,), dtype=torch.long)
    )
    x = torch.tensor(list(zip(shp, clr, pos)), dtype=torch.long)
    y = torch.tensor([label2i[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


train_graphs = [
    seq_to_graph(ex["sequence"], ex["label"]) for ex in spr_dataset["train"]
]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr_dataset["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in spr_dataset["test"]]


# ────────────────── MODEL ─────────────────────────────────────
class SPR_RGCN(nn.Module):
    def __init__(self, emb_dim=32, hidden=128):
        super().__init__()
        self.shape_emb = nn.Embedding(NUM_SH, emb_dim)
        self.color_emb = nn.Embedding(NUM_CL, emb_dim)
        self.pos_emb = nn.Embedding(MAX_POS, emb_dim)
        in_dim = emb_dim * 3
        self.conv1 = RGCNConv(in_dim, hidden, num_relations=3)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=3)
        self.cls = nn.Linear(hidden, NUM_LB)

    def forward(self, data):
        x = torch.cat(
            [
                self.shape_emb(data.x[:, 0]),
                self.color_emb(data.x[:, 1]),
                self.pos_emb(data.x[:, 2].clamp(max=MAX_POS - 1)),
            ],
            dim=-1,
        )
        h = self.conv1(x, data.edge_index, data.edge_type).relu()
        h = self.conv2(h, data.edge_index, data.edge_type).relu()
        hg = global_mean_pool(h, data.batch)  # graph-level representation
        return self.cls(hg)  # FIX: use pooled embedding


# ────────────────── TRAINING LOOP ─────────────────────────────
def train_model(epochs=15, batch_size=64, lr=1e-3, patience=4):
    model = SPR_RGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    tr_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_graphs, batch_size=batch_size)

    best_val_loss, wait = float("inf"), 0
    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(tr_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss, ys, ps, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss += criterion(out, batch.y).item() * batch.num_graphs
                ps.extend(out.argmax(1).cpu().tolist())
                ys.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq)
        val_loss /= len(val_loader.dataset)
        cwa = color_weighted_accuracy(seqs, ys, ps)
        swa = shape_weighted_accuracy(seqs, ys, ps)
        dwa = 0.5 * (cwa + swa)

        # ---- Logging ----
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"CWA={cwa:.3f}  SWA={swa:.3f}  DWA={dwa:.3f}"
        )
        experiment_data["SPR"]["losses"]["train"].append(train_loss)
        experiment_data["SPR"]["losses"]["val"].append(val_loss)
        experiment_data["SPR"]["metrics"]["val"].append(
            {"CWA": cwa, "SWA": swa, "DWA": dwa}
        )

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss, wait = val_loss, 0
            torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # ---- Testing ----
    model.load_state_dict(
        torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
    )
    model.eval()
    tst_loader = DataLoader(test_graphs, batch_size=128)
    ys, ps, seqs = [], [], []
    with torch.no_grad():
        for batch in tst_loader:
            batch = batch.to(device)
            out = model(batch)
            ps.extend(out.argmax(1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
            seqs.extend(batch.seq)
    cwa = color_weighted_accuracy(seqs, ys, ps)
    swa = shape_weighted_accuracy(seqs, ys, ps)
    dwa = 0.5 * (cwa + swa)
    experiment_data["SPR"]["predictions"] = ps
    experiment_data["SPR"]["ground_truth"] = ys
    experiment_data["SPR"]["metrics"]["test"] = {"CWA": cwa, "SWA": swa, "DWA": dwa}
    print("TEST METRICS:", experiment_data["SPR"]["metrics"]["test"])


# ────────────────── RUN ALL ───────────────────────────────────
start_time = time.time()
train_model()
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved all experiment data.")
print("Total elapsed:", round(time.time() - start_time, 2), "s")
