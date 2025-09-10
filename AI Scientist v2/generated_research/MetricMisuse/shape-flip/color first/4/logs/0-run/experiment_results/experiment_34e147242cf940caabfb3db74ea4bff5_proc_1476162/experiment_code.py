import os, random, string, pathlib, numpy as np, torch, time
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from datasets import load_dataset, DatasetDict
from typing import List, Tuple


# -------------------- misc helpers --------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(0)


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(_w if t == p else 0 for _w, t, p in zip(w, y_true, y_pred)) / max(
        1e-6, sum(w)
    )


# -------------------- data --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    return DatasetDict(
        {
            sp: load_dataset(
                "csv",
                data_files=str(root / f"{sp}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )
            for sp in ["train", "dev", "test"]
        }
    )


def generate_synthetic_dataset(n: int) -> Tuple[List[str], List[int]]:
    shapes = list(string.ascii_uppercase[:5])
    colors = list("12345")
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
    print("Loaded real SPR_BENCH data")
except Exception as e:
    print("Using synthetic data:", e)
    tr_s, tr_y = generate_synthetic_dataset(500)
    dv_s, dv_y = generate_synthetic_dataset(120)
    ts_s, ts_y = generate_synthetic_dataset(120)
    spr = DatasetDict(
        {
            "train": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
            "dev": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
            "test": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
        }
    )
    spr["train"] = spr["train"].add_column("sequence", tr_s).add_column("label", tr_y)
    spr["dev"] = spr["dev"].add_column("sequence", dv_s).add_column("label", dv_y)
    spr["test"] = spr["test"].add_column("sequence", ts_s).add_column("label", ts_y)


# -------------------- vocab --------------------
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


shape2idx, color2idx, label2idx = build_vocabs(spr["train"])
num_shapes, len_colors, len_labels = len(shape2idx), len(color2idx), len(label2idx)


# -------------------- graph conversion --------------------
def sequence_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    n = len(toks)
    shp = [shape2idx[t[0]] for t in toks]
    col = [color2idx[t[1:]] for t in toks]
    x = torch.tensor(list(zip(shp, col)), dtype=torch.long)
    src, dst = [], []
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


train_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]


# -------------------- model --------------------
class SPRGraphNet(nn.Module):
    def __init__(self, ns, nc, ncls, emb=16, hid=32):
        super().__init__()
        self.shape_emb = nn.Embedding(ns, emb)
        self.color_emb = nn.Embedding(nc, emb)
        self.g1 = SAGEConv(emb * 2, hid)
        self.g2 = SAGEConv(hid, hid)
        self.cls = nn.Linear(hid, ncls)

    def forward(self, data):
        h = torch.cat(
            [self.shape_emb(data.x[:, 0]), self.color_emb(data.x[:, 1])], dim=-1
        )
        h = self.g1(h, data.edge_index).relu()
        h = self.g2(h, data.edge_index).relu()
        return self.cls(global_mean_pool(h, data.batch))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------- hyperparameter search --------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
experiment_data = {
    "weight_decay": {
        "SPR_BENCH": {
            "hyperparams": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    set_seed(0)  # fresh start
    model = SPRGraphNet(num_shapes, len_colors, len_labels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=64)

    epochs = 5
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tr_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * batch.num_graphs
        tr_loss /= len(train_loader.dataset)

        # val
        model.eval()
        val_loss = 0
        ys = []
        preds = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = crit(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds += out.argmax(-1).cpu().tolist()
                ys += batch.y.cpu().tolist()
                seqs += batch.seq
        val_loss /= len(dev_loader.dataset)
        cwa2_val = complexity_weighted_accuracy(seqs, ys, preds)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  CWA2={cwa2_val:.4f}"
        )

    # final metrics on train
    model.eval()
    tr_seq, tr_ys, tr_preds = [], [], []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            o = model(batch)
            tr_preds += o.argmax(-1).cpu().tolist()
            tr_ys += batch.y.cpu().tolist()
            tr_seq += batch.seq
    cwa2_train = complexity_weighted_accuracy(tr_seq, tr_ys, tr_preds)

    # store
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
    ed["hyperparams"].append({"weight_decay": wd})
    ed["metrics"]["train"].append(cwa2_train)
    ed["metrics"]["val"].append(cwa2_val)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["predictions"].append(preds)
    ed["ground_truth"].append(ys)

# -------------------- save --------------------
np.save("experiment_data.npy", experiment_data)
print("\nSaved results to experiment_data.npy")
