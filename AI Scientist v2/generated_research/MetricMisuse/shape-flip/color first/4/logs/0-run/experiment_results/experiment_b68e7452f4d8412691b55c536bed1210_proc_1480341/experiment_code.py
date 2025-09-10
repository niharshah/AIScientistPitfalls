import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------- helper metrics
def count_color_variety(seq):  # token[1:] is colour id
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1e-9, sum(w))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1e-9, sum(w))


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1e-9, sum(w))


# --------------------------------------------------------------------- dataset utils
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def generate_synth(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labs = [], []
    for _ in range(n):
        L = random.randint(5, 12)
        seqs.append(
            " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        )
        labs.append(random.randint(0, 2))
    return seqs, labs


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    dsets = load_spr_bench(data_root)
except Exception as e:
    print("Falling back to synthetic data:", e)
    tr_s, tr_l = generate_synth(1000)
    dv_s, dv_l = generate_synth(300)
    ts_s, ts_l = generate_synth(300)
    dummy = load_dataset(
        "json", data_files={"train": [{}]}, split="train"
    ).remove_columns([])
    dsets = DatasetDict(
        {
            "train": dummy.add_column("sequence", tr_s).add_column("label", tr_l),
            "dev": dummy.add_column("sequence", dv_s).add_column("label", dv_l),
            "test": dummy.add_column("sequence", ts_s).add_column("label", ts_l),
        }
    )

max_train = 8000
if len(dsets["train"]) > max_train:
    dsets["train"] = dsets["train"].shuffle(seed=42).select(range(max_train))


# --------------------------------------------------------------------- vocab build
def build_vocabs(dset):
    shapes, colors, labels = set(), set(), set()
    for ex in dset:
        for tok in ex["sequence"].split():
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
        {l: i for i, l in enumerate(sorted(labels))},
    )


shape2i, color2i, label2i = build_vocabs(dsets["train"])
num_shapes, num_colors, num_labels = len(shape2i), len(color2i), len(label2i)
max_pos = 20


# --------------------------------------------------------------------- seq -> graph
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shp = [shape2i[t[0]] for t in toks]
    col = [color2i[t[1:]] for t in toks]
    pos = list(range(n))

    src, dst, etype = [], [], []
    # relation 0: next-token (bidirectional)
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etype.extend([0, 0])
    # relation 1: same-colour
    col_groups = {}
    for i, c in enumerate(col):
        col_groups.setdefault(c, []).append(i)
    for g in col_groups.values():
        for i in g:
            for j in g:
                if i != j:
                    src.append(i)
                    dst.append(j)
                    etype.append(1)
    # relation 2: same-shape
    shp_groups = {}
    for i, s in enumerate(shp):
        shp_groups.setdefault(s, []).append(i)
    for g in shp_groups.values():
        for i in g:
            for j in g:
                if i != j:
                    src.append(i)
                    dst.append(j)
                    etype.append(2)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    x = torch.tensor(list(zip(shp, col, pos)), dtype=torch.long)
    y = torch.tensor([label2i[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


train_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in dsets["train"]]
dev_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in dsets["dev"]]
test_graphs = [seq_to_graph(ex["sequence"], ex["label"]) for ex in dsets["test"]]


# --------------------------------------------------------------------- model
class SPR_RGCN(nn.Module):
    def __init__(
        self, n_shape, n_color, n_pos, n_class, emb_dim=32, hid=64, num_rel=3, drop=0.2
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, emb_dim)
        self.color_emb = nn.Embedding(n_color, emb_dim)
        self.pos_emb = nn.Embedding(n_pos, emb_dim // 2)
        in_dim = emb_dim * 2 + emb_dim // 2
        self.conv1 = RGCNConv(in_dim, hid, num_relations=num_rel)
        self.conv2 = RGCNConv(hid, hid, num_relations=num_rel)
        self.lin = nn.Linear(hid, n_class)
        self.drop = nn.Dropout(drop)

    def forward(self, data):
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        pos = self.pos_emb(data.x[:, 2])
        h = torch.cat([shp, col, pos], dim=-1)
        h = self.conv1(h, data.edge_index, data.edge_type).relu()
        h = self.drop(h)
        h = self.conv2(h, data.edge_index, data.edge_type).relu()
        hg = global_mean_pool(h, data.batch)
        return self.lin(hg)


model = SPR_RGCN(num_shapes, num_colors, max_pos, num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# loaders
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)

# --------------------------------------------------------------------- tracking dict
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------------------------------------------------------------- train loop
num_epochs = 12
best_val_loss = 1e9
for epoch in range(1, num_epochs + 1):
    # ---------- train
    model.train()
    tot_loss, s_tr, y_tr, p_tr = 0, [], [], []
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
        s_tr.extend(batch.seq)
    train_loss = tot_loss / len(train_loader.dataset)
    train_comp = comp_weighted_accuracy(s_tr, y_tr, p_tr)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((epoch, train_comp))

    # ---------- validation
    model.eval()
    v_loss, s_v, y_v, p_v = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            v_loss += criterion(out, batch.y).item() * batch.num_graphs
            preds = out.argmax(dim=-1).cpu().tolist()
            p_v.extend(preds)
            y_v.extend(batch.y.cpu().tolist())
            s_v.extend(batch.seq)
    val_loss = v_loss / len(dev_loader.dataset)
    val_comp = comp_weighted_accuracy(s_v, y_v, p_v)
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, val_comp))
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CompWA_val={val_comp:.4f}"
    )

    # simple early stop save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))

# --------------------------------------------------------------------- test evaluation
model.load_state_dict(torch.load(os.path.join(working_dir, "best_model.pt")))
model.eval()
s_t, y_t, p_t = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds = out.argmax(dim=-1).cpu().tolist()
        p_t.extend(preds)
        y_t.extend(batch.y.cpu().tolist())
        s_t.extend(batch.seq)

cwa = color_weighted_accuracy(s_t, y_t, p_t)
swa = shape_weighted_accuracy(s_t, y_t, p_t)
comp = comp_weighted_accuracy(s_t, y_t, p_t)
print(f"TEST  -> CWA={cwa:.4f} | SWA={swa:.4f} | CompWA={comp:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["test"] = [cwa, swa, comp]
experiment_data["SPR_BENCH"]["predictions"] = p_t
experiment_data["SPR_BENCH"]["ground_truth"] = y_t

# --------------------------------------------------------------------- save
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
