import os, pathlib, time, copy, numpy as np, torch, torch.nn as nn
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---------------- mandatory work dir & device ------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- helper: locate dataset -----------------------
def locate_spr_bench() -> pathlib.Path:
    cands = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in cands:
        if p and (p / "train.csv").exists():
            print("SPR_BENCH found at", p.resolve())
            return p.resolve()
    raise FileNotFoundError("Put SPR_BENCH folder next to script or set SPR_DATA_PATH")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


# ------------------------- metrics -----------------------------------
def count_color_variety(seq: str):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_struct_complexity(seq: str):
    return len(set(tok for tok in seq.split()))


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)) / max(
        sum(w), 1
    )


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)) / max(
        sum(w), 1
    )


def strwa(seqs, y_t, y_p):
    w = [count_struct_complexity(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)) / max(
        sum(w), 1
    )


# ---------------------- token & graph utils --------------------------
def extract_tokens(seq):
    return seq.strip().split()


def build_vocab(dataset):
    tokens, labels = set(), set()
    for ex in dataset:
        tokens.update(extract_tokens(ex["sequence"]))
        labels.add(ex["label"])
    tok2i = {t: i + 1 for i, t in enumerate(sorted(tokens))}  # 0 = padding
    lab2i = {l: i for i, l in enumerate(sorted(labels))}
    return tok2i, lab2i


# ----------- Unidirectional-Edges graph construction (ablation) ------
def seq_to_graph(example, tok2i, lab2i):
    seq = example["sequence"]
    toks = extract_tokens(seq)
    n = len(toks)
    node_idx = torch.tensor([tok2i[t] for t in toks], dtype=torch.long).unsqueeze(-1)

    src, dst, etype = [], [], []  # directed edges
    # relation 0: sequential neighbours  (only forward i -> i+1)
    for i in range(n - 1):
        src.append(i)
        dst.append(i + 1)
        etype.append(0)

    # build maps for colors & shapes
    color_map, shape_map = {}, {}
    for i, t in enumerate(toks):
        shape, color = t[0], t[1]
        color_map.setdefault(color, []).append(i)
        shape_map.setdefault(shape, []).append(i)

    # relation 1: same color (anchor -> peer)
    for idxs in color_map.values():
        anchor = idxs[0]
        for j in idxs[1:]:
            src.append(anchor)
            dst.append(j)
            etype.append(1)

    # relation 2: same shape (anchor -> peer)
    for idxs in shape_map.values():
        anchor = idxs[0]
        for j in idxs[1:]:
            src.append(anchor)
            dst.append(j)
            etype.append(2)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    y = torch.tensor([lab2i[example["label"]]], dtype=torch.long)
    g = Data(x=node_idx, edge_index=edge_index, edge_type=edge_type, y=y)
    g.seq = seq
    return g


# ---------------------- load data ------------------------------------
DATA_PATH = locate_spr_bench()
ds = load_spr(DATA_PATH)
token2idx, label2idx = build_vocab(ds["train"])
idx2label = {v: k for k, v in label2idx.items()}

train_graphs = [seq_to_graph(ex, token2idx, label2idx) for ex in ds["train"]]
dev_graphs = [seq_to_graph(ex, token2idx, label2idx) for ex in ds["dev"]]
test_graphs = [seq_to_graph(ex, token2idx, label2idx) for ex in ds["test"]]

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)


# ----------------------- model ---------------------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, vocab, emb=64, hid=96, classes=10, relations=3):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, emb, padding_idx=0)
        self.conv1 = RGCNConv(emb, hid, num_relations=relations)
        self.conv2 = RGCNConv(hid, hid, num_relations=relations)
        self.lin = nn.Linear(hid, classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.embed(x.squeeze(-1))
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        x = global_mean_pool(x, batch)
        x = self.drop(x)
        return self.lin(x)


# ------------------- training / evaluation helpers -------------------
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, preds, labels, seqs = 0.0, [], [], []
    for bt in loader:
        bt = bt.to(device)
        out = model(bt.x, bt.edge_index, bt.edge_type, bt.batch)
        loss = criterion(out, bt.y.view(-1))
        tot_loss += loss.item() * bt.num_graphs
        pr = out.argmax(-1).cpu().tolist()
        lb = bt.y.view(-1).cpu().tolist()
        preds.extend(pr)
        labels.extend(lb)
        seqs.extend(bt.seq)
    c = cwa(seqs, labels, preds)
    s = swa(seqs, labels, preds)
    r = strwa(seqs, labels, preds)
    bwa = (c + s) / 2.0
    return tot_loss / len(loader.dataset), bwa, c, s, r, preds, labels, seqs


# ------------------- experiment container ----------------------------
experiment_data = {
    "unidirectional_edges": {
        "spr_bench": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# -------------------------- training loop ----------------------------
model = SPR_RGCN(len(token2idx), classes=len(label2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
best_bwa, best_state, wait = -1, None, 0
max_epochs, patience = 20, 3

for epoch in range(1, max_epochs + 1):
    model.train()
    ep_loss = 0.0
    for bt in train_loader:
        bt = bt.to(device)
        optimizer.zero_grad()
        out = model(bt.x, bt.edge_index, bt.edge_type, bt.batch)
        loss = criterion(out, bt.y.view(-1))
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * bt.num_graphs
    train_loss = ep_loss / len(train_loader.dataset)

    val_loss, val_bwa, val_c, val_s, val_r, _, _, _ = evaluate(model, dev_loader)
    train_loss_eval, tr_bwa, tr_c, tr_s, tr_r, _, _, _ = evaluate(model, train_loader)

    # logging
    ed = experiment_data["unidirectional_edges"]["spr_bench"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append(
        {"BWA": tr_bwa, "CWA": tr_c, "SWA": tr_s, "StrWA": tr_r}
    )
    ed["metrics"]["val"].append(
        {"BWA": val_bwa, "CWA": val_c, "SWA": val_s, "StrWA": val_r}
    )
    ed["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"BWA={val_bwa:.4f}  CWA={val_c:.4f}  SWA={val_s:.4f}  StrWA={val_r:.4f}"
    )

    if val_bwa > best_bwa:
        best_bwa, best_state, wait = val_bwa, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping!")
            break

# --------------------------- test ------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_bwa, test_c, test_s, test_r, preds, labels, seqs = evaluate(
    model, test_loader
)
ed["predictions"] = preds
ed["ground_truth"] = labels
ed["test_metrics"] = {
    "loss": test_loss,
    "BWA": test_bwa,
    "CWA": test_c,
    "SWA": test_s,
    "StrWA": test_r,
}
print(
    f"TEST -> loss {test_loss:.4f}  BWA {test_bwa:.4f}  CWA {test_c:.4f}  SWA {test_s:.4f}  StrWA {test_r:.4f}"
)

# --------------------------- save ------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Experiment data saved.")
