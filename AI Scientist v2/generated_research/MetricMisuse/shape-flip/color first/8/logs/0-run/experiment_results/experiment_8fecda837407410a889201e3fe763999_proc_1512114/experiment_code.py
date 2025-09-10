import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- paths / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():
    print("SPR_BENCH not found – creating synthetic tiny data.")
    DATA_PATH.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split, size in [("train", 200), ("dev", 40), ("test", 40)]:
        rows = [["id", "sequence", "label"]]
        for i in range(size):
            n = rng.integers(3, 7)
            seq = " ".join(
                rng.choice(list("ABC")) + rng.choice(list("123")) for _ in range(n)
            )
            rows.append([f"{split}_{i}", seq, rng.choice(["yes", "no"])])
        import csv, itertools

        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------- vocab & label mapping ----------
def parse_token(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


shapes, colours = set(), set()
for r in dsets["train"]:
    for tok in r["sequence"].split():
        s, c = parse_token(tok)
        shapes.add(s)
        colours.add(c)
shape2id = {s: i for i, s in enumerate(sorted(shapes))}
col2id = {c: i for i, c in enumerate(sorted(colours))}
label2id = {l: i for i, l in enumerate(sorted({r["label"] for r in dsets["train"]}))}


# ---------- seq -> pyg graph ----------
def seq_to_graph(sequence, lbl):
    toks = sequence.split()
    n = len(toks)
    x = []
    for tok in toks:
        s, c = parse_token(tok)
        vec = np.zeros(len(shape2id) + len(col2id), dtype=np.float32)
        vec[shape2id[s]] = 1.0
        vec[len(shape2id) + col2id[c]] = 1.0
        x.append(vec)
    x = torch.tensor(np.stack(x))
    if n > 1:
        src = torch.arange(n - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label2id[lbl]]))


graph_train = [seq_to_graph(r["sequence"], r["label"]) for r in dsets["train"]]
graph_dev = [seq_to_graph(r["sequence"], r["label"]) for r in dsets["dev"]]
graph_test = [seq_to_graph(r["sequence"], r["label"]) for r in dsets["test"]]

# ---------- data loaders ----------
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)


# ---------- model ----------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64, num_classes=len(label2id)):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------- utils ----------
def drop_edges(edge_index, drop_prob):
    if drop_prob <= 0:
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_prob
    return edge_index[:, mask]


def complexity_weight(seq):
    toks = seq.split()
    return len({t[0] for t in toks}) + len(
        {(t[1:] if len(t) > 1 else "0") for t in toks}
    )


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


# ---------- experiment data ----------
experiment_data = {"edge_dropout": {}}

# ---------- sweep ----------
drop_rates = [0.0, 0.1, 0.2, 0.3]
EPOCHS = 10
in_dim = len(shape2id) + len(col2id)

for p in drop_rates:
    tag = f"p_{p}"
    experiment_data["edge_dropout"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = GCN(in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ---- training ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = tot_corr = tot_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            ei = drop_edges(batch.edge_index, p)
            out = model(batch.x, ei, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
            tot_corr += (out.argmax(-1) == batch.y).sum().item()
            tot_n += batch.num_graphs
        tr_loss = tot_loss / tot_n
        tr_acc = tot_corr / tot_n
        experiment_data["edge_dropout"][tag]["losses"]["train"].append(tr_loss)
        experiment_data["edge_dropout"][tag]["metrics"]["train"].append(tr_acc)
        # ---- validation ----
        model.eval()
        v_loss = v_corr = v_n = 0
        preds = []
        gts = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                v_loss += loss.item() * batch.num_graphs
                pr = out.argmax(-1)
                preds.extend(pr.cpu().tolist())
                gts.extend(batch.y.cpu().tolist())
                v_corr += (pr == batch.y).sum().item()
                v_n += batch.num_graphs
        val_loss = v_loss / v_n
        val_acc = v_corr / v_n
        experiment_data["edge_dropout"][tag]["losses"]["val"].append(val_loss)
        experiment_data["edge_dropout"][tag]["metrics"]["val"].append(val_acc)
        if epoch == EPOCHS:  # store final preds
            experiment_data["edge_dropout"][tag]["predictions"] = preds
            experiment_data["edge_dropout"][tag]["ground_truth"] = gts
        print(f"[p={p}] epoch {epoch}: val_loss={val_loss:.4f}")
    # ---- Comp-Weighted Acc on dev ----
    seqs = [r["sequence"] for r in dsets["dev"]]
    cwa = comp_weighted_accuracy(seqs, gts, preds)
    experiment_data["edge_dropout"][tag]["comp_weighted_acc"] = cwa
    print(f"[p={p}] Complexity-Weighted Acc (dev): {cwa:.4f}")

# ---------- save & plot ----------
for tag in experiment_data["edge_dropout"]:
    plt.figure()
    plt.plot(experiment_data["edge_dropout"][tag]["losses"]["train"], label="train")
    plt.plot(experiment_data["edge_dropout"][tag]["losses"]["val"], label="val")
    plt.title(f"Loss (edge_dropout={tag})")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{tag}.png"))
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All results saved to ./working")
