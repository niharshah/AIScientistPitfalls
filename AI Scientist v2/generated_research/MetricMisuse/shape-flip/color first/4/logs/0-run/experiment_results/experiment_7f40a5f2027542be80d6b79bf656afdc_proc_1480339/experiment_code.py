import os, random, string, pathlib, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# --------- mandatory working dir ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- device handling ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- utility metrics -----------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1e-9, sum(w))


# ---------------- data loading --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def generate_synth(n: int):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(random.randint(0, 2))
    return seqs, labels


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    dsets = load_spr_bench(data_root)
except Exception as e:  # fallback small synthetic set
    print("SPR_BENCH not found, using synthetic tiny set.", e)
    tr_seq, tr_y = generate_synth(1200)
    dv_seq, dv_y = generate_synth(300)
    ts_seq, ts_y = generate_synth(300)
    dummy = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    dsets = DatasetDict(
        {
            "train": dummy.add_column("sequence", tr_seq).add_column("label", tr_y),
            "dev": dummy.add_column("sequence", dv_seq).add_column("label", dv_y),
            "test": dummy.add_column("sequence", ts_seq).add_column("label", ts_y),
        }
    )

# optional small subset for speed
max_train = 5000
if len(dsets["train"]) > max_train:
    dsets["train"] = dsets["train"].shuffle(seed=42).select(range(max_train))


# -------------- vocab construction ----------------
def build_vocabs(dataset):
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


shape2idx, color2idx, label2idx = build_vocabs(dsets["train"])
num_shapes, len_colors, len_labels = len(shape2idx), len(color2idx), len(label2idx)


# -------------- graph construction ----------------
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shape_idx = [shape2idx[t[0]] for t in toks]
    color_idx = [color2idx[t[1:]] for t in toks]
    pos_idx = list(range(n))
    # sequential edges
    src, dst = [], []
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    # colour edges
    col_groups = {}
    for i, c in enumerate(color_idx):
        col_groups.setdefault(c, []).append(i)
    for group in col_groups.values():
        for i in group:
            for j in group:
                if i != j:
                    src.append(i)
                    dst.append(j)
    # shape edges
    shp_groups = {}
    for i, s in enumerate(shape_idx):
        shp_groups.setdefault(s, []).append(i)
    for group in shp_groups.values():
        for i in group:
            for j in group:
                if i != j:
                    src.append(i)
                    dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(list(zip(shape_idx, color_idx, pos_idx)), dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in dsets["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in dsets["dev"]]


# -------------- model -------------------------------------
class SPRGNN(nn.Module):
    def __init__(self, n_shape, n_color, max_pos, n_class, emb=32, hid=64, drop=0.2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb)
        self.color_emb = nn.Embedding(n_color, emb)
        self.pos_emb = nn.Embedding(max_pos, emb // 2)
        in_dim = emb * 2 + emb // 2
        self.g1 = SAGEConv(in_dim, hid)
        self.g2 = SAGEConv(hid, hid)
        self.dropout = nn.Dropout(drop)
        self.cls = nn.Linear(hid, n_class)

    def forward(self, data):
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        pos = self.pos_emb(data.x[:, 2])
        h = torch.cat([shp, col, pos], dim=-1)
        h = self.g1(h, data.edge_index).relu()
        h = self.dropout(h)
        h = self.g2(h, data.edge_index).relu()
        hg = global_mean_pool(h, data.batch)
        return self.cls(hg)


max_pos = 20  # sequences are short
model = SPRGNN(num_shapes, len_colors, max_pos, len_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -------------- loaders -----------------------------------
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)

# -------------- experiment tracking -----------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------- training ----------------------------------
num_epochs_grid = [8, 16]
for num_epochs in num_epochs_grid:
    print(f"\n=== Training for {num_epochs} epochs ===")
    # re-init model for each run
    model.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        tot_loss, seqs_tr, y_tr, p_tr = 0, [], [], []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=-1).cpu().tolist()
            p_tr.extend(preds)
            y_tr.extend(batch.y.cpu().tolist())
            seqs_tr.extend(batch.seq)
        train_loss = tot_loss / len(train_loader.dataset)
        train_compwa = comp_weighted_accuracy(seqs_tr, y_tr, p_tr)
        # ---- validate ----
        model.eval()
        vloss, seqs_v, y_v, p_v = 0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                vloss += criterion(out, batch.y).item() * batch.num_graphs
                preds = out.argmax(dim=-1).cpu().tolist()
                p_v.extend(preds)
                y_v.extend(batch.y.cpu().tolist())
                seqs_v.extend(batch.seq)
        val_loss = vloss / len(dev_loader.dataset)
        val_compwa = comp_weighted_accuracy(seqs_v, y_v, p_v)
        # ---- logging ----
        experiment_data["SPR_BENCH"]["losses"]["train"].append(
            (num_epochs, epoch, train_loss)
        )
        experiment_data["SPR_BENCH"]["losses"]["val"].append(
            (num_epochs, epoch, val_loss)
        )
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(
            (num_epochs, epoch, train_compwa)
        )
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            (num_epochs, epoch, val_compwa)
        )
        print(
            f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}  CompWA_val={val_compwa:.4f}"
        )
    # keep last val predictions
    experiment_data["SPR_BENCH"]["predictions"].append((num_epochs, p_v))
    experiment_data["SPR_BENCH"]["ground_truth"] = y_v  # same for all runs

# -------------- Save experiment ---------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
