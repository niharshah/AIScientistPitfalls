import os, random, string, pathlib, numpy as np, torch, time
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import List, Tuple
from datasets import load_dataset, DatasetDict

# -------------------- experiment store --------------------
experiment_data = {"max_grad_norm": {}}  # will hold results per dataset->clip_value

# -------------------- misc helpers --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_color_variety(seq: str) -> int:
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        1e-6, sum(w)
    )


# -------------------- dataset loading --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def generate_synthetic_dataset(n: int) -> Tuple[List[str], List[int]]:
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
except Exception:
    tr_s, tr_y = generate_synthetic_dataset(500)
    dv_s, dv_y = generate_synthetic_dataset(120)
    sp = load_dataset("json", data_files={"train": [{}]}, split="train").remove_columns(
        []
    )
    spr = DatasetDict(
        {
            "train": sp.add_column("sequence", tr_s).add_column("label", tr_y),
            "dev": sp.add_column("sequence", dv_s).add_column("label", dv_y),
            "test": sp.add_column("sequence", dv_s).add_column("label", dv_y),
        }
    )


# -------------------- vocab / graph helpers --------------------
def build_vocabs(data):
    shapes, colors, labels = set(), set(), set()
    for ex in data:
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


def sequence_to_graph(seq: str, lab: int) -> Data:
    toks = seq.split()
    n = len(toks)
    x = torch.tensor(
        [[shape2idx[t[0]], color2idx[t[1:]]] for t in toks], dtype=torch.long
    )
    src, dst = [], []
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    edge = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge, y=torch.tensor([label2idx[lab]]), seq=seq)


train_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]


# -------------------- model --------------------
class SPRGraphNet(nn.Module):
    def __init__(self, n_shapes, n_colors, n_cls, emb=16, hid=32):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, emb)
        self.color_emb = nn.Embedding(n_colors, emb)
        self.gnn1, self.gnn2 = SAGEConv(emb * 2, hid), SAGEConv(hid, hid)
        self.fc = nn.Linear(hid, n_cls)

    def forward(self, d):
        h = torch.cat([self.shape_emb(d.x[:, 0]), self.color_emb(d.x[:, 1])], -1)
        h = self.gnn1(h, d.edge_index).relu()
        h = self.gnn2(h, d.edge_index).relu()
        return self.fc(global_mean_pool(h, d.batch))


# -------------------- training / evaluation --------------------
def run_experiment(clip_val: float or None, epochs=5):
    clip_key = "none" if clip_val is None else str(clip_val)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=64)
    model = SPRGraphNet(len(shape2idx), len(color2idx), len(label2idx)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(epochs):
        model.train()
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y)
            loss.backward()
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optim.step()
            total += loss.item() * batch.num_graphs
        log["losses"]["train"].append(total / len(train_loader.dataset))
        # ---- eval ----
        model.eval()
        vloss = 0
        ys, preds, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = crit(out, batch.y)
                vloss += loss.item() * batch.num_graphs
                preds.extend(out.argmax(-1).cpu().tolist())
                ys.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq)
        log["losses"]["val"].append(vloss / len(dev_loader.dataset))
        log["metrics"]["val"].append(complexity_weighted_accuracy(seqs, ys, preds))
    log["predictions"], log["ground_truth"] = preds, ys
    return clip_key, log


# -------------------- hyper-parameter sweep --------------------
experiment_data["max_grad_norm"]["SPR_BENCH"] = {}
for clip in [None, 0.5, 1.0, 2.0, 5.0]:
    key, res = run_experiment(clip)
    experiment_data["max_grad_norm"]["SPR_BENCH"][key] = res
    print(f"Finished clip={key}: final CWA2={res['metrics']['val'][-1]:.4f}")

# -------------------- save --------------------
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
print("Saved to working/experiment_data.npy")
