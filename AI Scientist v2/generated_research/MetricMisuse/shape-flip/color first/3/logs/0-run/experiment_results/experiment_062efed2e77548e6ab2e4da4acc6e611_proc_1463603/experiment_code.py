import os, pathlib, random, time, copy, numpy as np, torch, torch.nn as nn
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ------------ mandatory working dir ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ device -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------ experiment bookkeeping --------------------------------
experiment_data = {
    "spr_bench": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "test_metrics": {},
    }
}


# ------------ synthetic data fallback -------------------------------
def _write_dummy_csv(path: pathlib.Path, n_rows: int):
    shapes = ["A", "B", "C"]
    colors = ["x", "y", "z"]
    labels = ["L0", "L1", "L2"]
    with path.open("w") as f:
        f.write("id,sequence,label\n")
        for idx in range(n_rows):
            seq_len = random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
            )
            label = random.choice(labels)
            f.write(f"{idx},{seq},{label}\n")


def _make_dummy_bench(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    _write_dummy_csv(root / "train.csv", 300)
    _write_dummy_csv(root / "dev.csv", 60)
    _write_dummy_csv(root / "test.csv", 60)
    print("Dummy SPR_BENCH created at", root.resolve())


# ------------ locate dataset ----------------------------------------
def locate_spr_bench() -> pathlib.Path:
    cands = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in cands:
        if p and (p / "train.csv").exists():
            print("SPR_BENCH found at", p.resolve())
            return p.resolve()
    # not found -> create dummy
    dummy_path = pathlib.Path(working_dir) / "SPR_BENCH_DUMMY"
    _make_dummy_bench(dummy_path)
    return dummy_path.resolve()


# ------------ dataset loading ---------------------------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


# ------------ metrics -----------------------------------------------
def _uniq_colors(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def _uniq_shapes(seq):
    return len(set(t[0] for t in seq.split() if t))


def _weight_acc(seqs, y_true, y_pred, w_func):
    w = [w_func(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_acc(seqs, y_t, y_p):
    return _weight_acc(seqs, y_t, y_p, _uniq_colors)


def shape_weighted_acc(seqs, y_t, y_p):
    return _weight_acc(seqs, y_t, y_p, _uniq_shapes)


def scwa(seqs, y_t, y_p):  # Sequence-Complexity Weighted Accuracy
    weight = lambda s: _uniq_colors(s) * _uniq_shapes(s)
    return _weight_acc(seqs, y_t, y_p, weight)


# ------------ graph helpers -----------------------------------------
def extract_tokens(seq):
    return seq.strip().split()


def build_vocab(dataset):
    tokens, labels = set(), set()
    for ex in dataset:
        tokens.update(extract_tokens(ex["sequence"]))
        labels.add(ex["label"])
    tok2i = {t: i + 1 for i, t in enumerate(sorted(tokens))}  # 0 reserved
    lab2i = {l: i for i, l in enumerate(sorted(labels))}
    return tok2i, lab2i


def seq_to_graph(example, tok2i, lab2i):
    toks = extract_tokens(example["sequence"])
    n = len(toks)
    x = torch.tensor([tok2i[t] for t in toks], dtype=torch.long).unsqueeze(-1)
    src, dst, etype = [], [], []
    # relation 0: sequential neighbours
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        etype += [0, 0]
    # relation 1/2: same color / same shape
    color_map, shape_map = {}, {}
    for i, tok in enumerate(toks):
        sh, co = tok[0], tok[1]
        color_map.setdefault(co, []).append(i)
        shape_map.setdefault(sh, []).append(i)
    for ids in color_map.values():
        a = ids[0]
        for j in ids[1:]:
            src += [a, j]
            dst += [j, a]
            etype += [1, 1]
    for ids in shape_map.values():
        a = ids[0]
        for j in ids[1:]:
            src += [a, j]
            dst += [j, a]
            etype += [2, 2]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    y = torch.tensor([lab2i[example["label"]]], dtype=torch.long)
    g = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    g.seq = example["sequence"]
    return g


# ------------ model --------------------------------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, vocab, emb=64, hid=96, classes=10, relations=3):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, emb, padding_idx=0)
        self.conv1 = RGCNConv(emb, hid, relations)
        self.conv2 = RGCNConv(hid, hid, relations)
        self.lin = nn.Linear(hid, classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.embed(x.squeeze(-1))
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        x = global_mean_pool(x, batch)
        x = self.drop(x)
        return self.lin(x)


# ------------ evaluation helper -------------------------------------
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
    cwa = color_weighted_acc(seqs, labels, preds)
    swa = shape_weighted_acc(seqs, labels, preds)
    scw = scwa(seqs, labels, preds)
    bwa = (cwa + swa) / 2.0
    return tot_loss / len(loader.dataset), bwa, cwa, swa, scw, preds, labels, seqs


# ------------ main routine ------------------------------------------
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

model = SPR_RGCN(len(token2idx), classes=len(label2idx)).to(device)
# freeze embedding for ablation
model.embed.weight.requires_grad_(False)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3
)

best_bwa, best_state, wait = -1, None, 0
max_epochs, patience = 10, 3

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
    val_loss, val_bwa, val_c, val_s, val_scw, *_ = evaluate(model, dev_loader)
    tr_loss_eval, tr_bwa, tr_c, tr_s, tr_scw, *_ = evaluate(model, train_loader)

    # bookkeeping
    ed = experiment_data["spr_bench"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append(
        {"BWA": tr_bwa, "CWA": tr_c, "SWA": tr_s, "SCWA": tr_scw}
    )
    ed["metrics"]["val"].append(
        {"BWA": val_bwa, "CWA": val_c, "SWA": val_s, "SCWA": val_scw}
    )
    ed["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"BWA={val_bwa:.4f}  CWA={val_c:.4f}  SWA={val_s:.4f}  SCWA={val_scw:.4f}"
    )

    if val_bwa > best_bwa:
        best_bwa, best_state, wait = val_bwa, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

# ------------ testing ------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_bwa, test_c, test_s, test_scw, preds, labels, seqs = evaluate(
    model, test_loader
)
ed["predictions"] = preds
ed["ground_truth"] = labels
ed["test_metrics"] = {
    "loss": test_loss,
    "BWA": test_bwa,
    "CWA": test_c,
    "SWA": test_s,
    "SCWA": test_scw,
}

print(
    f"TEST -> loss {test_loss:.4f}  BWA {test_bwa:.4f}  "
    f"CWA {test_c:.4f}  SWA {test_s:.4f}  SCWA {test_scw:.4f}"
)

# ------------ save results ------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Experiment data saved to working/experiment_data.npy")
