import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- hyper-parameter grid ----------
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
EPOCHS = 10
BATCH_TRAIN, BATCH_EVAL = 64, 128
torch.manual_seed(0)
np.random.seed(0)

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():
    print("SPR_BENCH not found – creating tiny synthetic data.")
    os.makedirs(DATA_PATH, exist_ok=True)
    for split, size in [("train", 200), ("dev", 40), ("test", 40)]:
        rng = np.random.default_rng(0)
        seqs, labels = [], []
        shapes, cols = "ABC", "123"
        for _ in range(size):
            n = rng.integers(3, 7)
            seq = " ".join(
                rng.choice(list(shapes)) + rng.choice(list(cols)) for _ in range(n)
            )
            labels.append(rng.choice(["yes", "no"]))
            seqs.append(seq)
        import csv

        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i, (s, l) in enumerate(zip(seqs, labels)):
                w.writerow([f"{split}_{i}", s, l])

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------- vocab ----------
def parse_token(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


shapes, colours = set(), set()
for row in dsets["train"]:
    for tok in row["sequence"].split():
        s, c = parse_token(tok)
        shapes.add(s)
        colours.add(c)
shape2id = {s: i for i, s in enumerate(sorted(shapes))}
col2id = {c: i for i, c in enumerate(sorted(colours))}
label2id = {l: i for i, l in enumerate(sorted({r["label"] for r in dsets["train"]}))}


# ---------- graph conversion ----------
def seq_to_graph(sequence, lbl):
    toks = sequence.split()
    n = len(toks)
    feats = []
    for tok in toks:
        s, c = parse_token(tok)
        vec = np.zeros(len(shape2id) + len(col2id), dtype=np.float32)
        vec[shape2id[s]] = 1.0
        vec[len(shape2id) + col2id[c]] = 1.0
        feats.append(vec)
    x = torch.tensor(np.stack(feats))
    if n > 1:
        src = torch.arange(n - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]])
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train, graph_dev, graph_test = map(build_graph_dataset, ["train", "dev", "test"])
train_loader = DataLoader(graph_train, batch_size=BATCH_TRAIN, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=BATCH_EVAL, shuffle=False)


# ---------- metrics ----------
def complexity_weight(seq):
    toks = seq.split()
    sh = {t[0] for t in toks}
    co = {t[1:] if len(t) > 1 else "0" for t in toks}
    return len(sh) + len(co)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


# ---------- model ----------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64, n_cls=len(label2id)):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, n_cls)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------- experiment container ----------
experiment_data = {"weight_decay": {"SPR_BENCH": {}}}

# ---------- training + evaluation for each weight_decay ----------
for wd in WEIGHT_DECAYS:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = GCN(len(shape2id) + len(col2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    key = f"wd_{wd}"
    experiment_data["weight_decay"]["SPR_BENCH"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [label2id[r["label"]] for r in dsets["dev"]],
    }

    # --- epochs ---
    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tot_loss = tot_corr = tot_ex = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
            tot_corr += int((out.argmax(-1) == batch.y).sum().item())
            tot_ex += batch.num_graphs
        tr_loss = tot_loss / tot_ex
        tr_acc = tot_corr / tot_ex

        # validate
        model.eval()
        v_loss = v_corr = v_ex = 0
        preds = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                v_loss += loss.item() * batch.num_graphs
                v_corr += int((out.argmax(-1) == batch.y).sum().item())
                v_ex += batch.num_graphs
                preds.extend(out.argmax(-1).cpu().tolist())
        val_loss = v_loss / v_ex
        val_acc = v_corr / v_ex

        # record
        d = experiment_data["weight_decay"]["SPR_BENCH"][key]
        d["losses"]["train"].append(tr_loss)
        d["losses"]["val"].append(val_loss)
        d["metrics"]["train"].append(tr_acc)
        d["metrics"]["val"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

    # --- final dev predictions & CompWA ---
    experiment_data["weight_decay"]["SPR_BENCH"][key]["predictions"] = preds
    compwa = comp_weighted_accuracy(
        [r["sequence"] for r in dsets["dev"]],
        experiment_data["weight_decay"]["SPR_BENCH"][key]["ground_truth"],
        preds,
    )
    print(f"Weight decay {wd}: Complexity-Weighted Accuracy (dev) = {compwa:.4f}")

# ---------- plot ----------
plt.figure(figsize=(6, 4))
for wd in WEIGHT_DECAYS:
    k = f"wd_{wd}"
    vals = experiment_data["weight_decay"]["SPR_BENCH"][k]["losses"]["val"]
    plt.plot(vals, label=k)
plt.title("Validation Loss vs. Weight Decay")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to working/experiment_data.npy and loss_curve.png")
