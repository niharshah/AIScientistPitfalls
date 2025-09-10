import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ------------ dirs & device ------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------ load or create SPR_BENCH ------------
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
    rng = np.random.default_rng(0)
    for split, s in [("train", 200), ("dev", 40), ("test", 40)]:
        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            import csv

            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(s):
                n = rng.integers(3, 7)
                seq = " ".join(
                    rng.choice(list("ABC")) + str(rng.integers(1, 4)) for _ in range(n)
                )
                w.writerow([f"{split}_{i}", seq, rng.choice(["yes", "no"])])
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ------------ vocab building ------------
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
print("Shapes", shape2id, "Colours", col2id, "Labels", label2id)

# ------------ seq -> graph ------------
from torch_geometric.data import Data


def seq_to_graph(sequence, lbl):
    toks = sequence.split()
    x = []
    for tok in toks:
        s, c = parse_token(tok)
        v = np.zeros(len(shape2id) + len(col2id), dtype=np.float32)
        v[shape2id[s]] = 1.0
        v[len(shape2id) + col2id[c]] = 1.0
        x.append(v)
    x = torch.tensor(np.stack(x))
    n = len(toks)
    if n > 1:
        src = torch.arange(0, n - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label2id[lbl]]))


def build(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train, graph_dev, graph_test = map(build, ["train", "dev", "test"])
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)


# ------------ model definition ------------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ------------ helper metric ------------
def complexity_weight(seq):
    toks = seq.split()
    sh = {t[0] for t in toks}
    co = {t[1:] if len(t) > 1 else "0" for t in toks}
    return len(sh) + len(co)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


# ------------ experiment container ------------
experiment_data = {"hidden_dim": {"SPR_BENCH": {}}}

hidden_dims = [32, 64, 128, 256]
EPOCHS = 10
input_dim = len(shape2id) + len(col2id)
seqs_dev = [r["sequence"] for r in dsets["dev"]]
gt_dev = [label2id[r["label"]] for r in dsets["dev"]]

for hid in hidden_dims:
    print(f"\n=== Hidden dim: {hid} ===")
    model = GCN(input_dim, hid, len(label2id)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_losses, tr_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(1, EPOCHS + 1):
        # --- training ---
        model.train()
        tl, tc, tex = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optim.step()
            tl += loss.item() * batch.num_graphs
            tc += int((out.argmax(-1) == batch.y).sum().item())
            tex += batch.num_graphs
        tr_losses.append(tl / tex)
        tr_accs.append(tc / tex)

        # --- validation ---
        model.eval()
        vl, vc, vex = 0.0, 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                vl += loss.item() * batch.num_graphs
                vc += int((out.argmax(-1) == batch.y).sum().item())
                vex += batch.num_graphs
        val_losses.append(vl / vex)
        val_accs.append(vc / vex)
        print(
            f"Epoch {epoch}: val_loss {val_losses[-1]:.4f}  val_acc {val_accs[-1]:.4f}"
        )

    # --- final dev predictions & CompWA ---
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            preds.extend(
                model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
            )
    compwa = comp_weighted_accuracy(seqs_dev, gt_dev, preds)
    print(f"Hidden {hid}: Comp-Weighted Acc = {compwa:.4f}")

    # --- save per-hid info ---
    hdict = {
        "metrics": {"train": tr_accs, "val": val_accs},
        "losses": {"train": tr_losses, "val": val_losses},
        "predictions": preds,
        "ground_truth": gt_dev,
        "comp_weighted_acc": compwa,
    }
    experiment_data["hidden_dim"]["SPR_BENCH"][str(hid)] = hdict

    # plot loss curve
    plt.figure()
    plt.plot(tr_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title(f"Loss curve (hid={hid})")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_hid{hid}.png"))
    plt.close()

# ------------ persist ------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All results saved in ./working/experiment_data.npy")
