import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ------------ required working dir -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ GPU/CPU handling -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------ helper: load SPR_BENCH -------------
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
if not DATA_PATH.exists():  # build tiny synthetic fallback
    print("SPR_BENCH not found – creating tiny synthetic data.")
    os.makedirs(DATA_PATH, exist_ok=True)
    for split, s in [("train", 200), ("dev", 40), ("test", 40)]:
        rng = np.random.default_rng(0)
        shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
        seqs, labels = [], []
        for _ in range(s):
            n = rng.integers(3, 7)
            seq = " ".join(rng.choice(shapes) + rng.choice(colors) for _ in range(n))
            labels.append(rng.choice(["yes", "no"]))
            seqs.append(seq)
        import csv

        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i, (seq, lbl) in enumerate(zip(seqs, labels)):
                w.writerow([f"{split}_{i}", seq, lbl])

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ------------ vocab -------------
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
print("Shapes:", shape2id, "\nColours:", col2id)


# ------------ sequence -> graph conversion -------------
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
        src = torch.arange(0, n - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train, graph_dev, graph_test = map(build_graph_dataset, ["train", "dev", "test"])
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)
test_loader = DataLoader(graph_test, batch_size=128, shuffle=False)


# ------------ dynamic GCN -------------
class GCNStack(torch.nn.Module):
    def __init__(self, in_dim, hid, num_layers, num_classes):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        if num_layers < 1:
            raise ValueError("num_layers must be >=1")
        self.convs.append(GCNConv(in_dim, hid))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hid, hid))
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ------------ metric helpers -------------
def complexity_weight(seq):
    toks = seq.split()
    shapes = {t[0] for t in toks}
    cols = {t[1:] if len(t) > 1 else "0" for t in toks}
    return len(shapes) + len(cols)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    return (
        sum(wi if a == b else 0 for wi, a, b in zip(w, y_true, y_pred)) / sum(w)
        if sum(w) > 0
        else 0.0
    )


# ------------ experiment data container -------------
experiment_data = {"num_gcn_layers": {}}

DEPTHS = [2, 3, 4]
EPOCHS = 10

for depth in DEPTHS:
    print(f"\n=== Training model with {depth} GCN layers ===")
    run_key = str(depth)
    experiment_data["num_gcn_layers"][run_key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = GCNStack(
        len(shape2id) + len(col2id), hid=64, num_layers=depth, num_classes=len(label2id)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- training loop ----
    for epoch in range(1, EPOCHS + 1):
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
        tr_loss, tr_acc = tot_loss / tot_ex, tot_corr / tot_ex
        ed = experiment_data["num_gcn_layers"][run_key]["SPR_BENCH"]
        ed["losses"]["train"].append(tr_loss)
        ed["metrics"]["train"].append(tr_acc)

        # ---- validation ----
        model.eval()
        v_loss = v_corr = v_ex = 0
        all_pred = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                v_loss += loss.item() * batch.num_graphs
                preds = out.argmax(-1).cpu()
                v_corr += int((preds == batch.y.cpu()).sum().item())
                v_ex += batch.num_graphs
                all_pred.extend(preds.tolist())
        val_loss, val_acc = v_loss / v_ex, v_corr / v_ex
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["val"].append(val_acc)
        print(
            f"Depth {depth} | Epoch {epoch} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}"
        )

    # ---- Complexity Weighted Acc on dev ----
    dev_seqs = [r["sequence"] for r in dsets["dev"]]
    y_true = [label2id[r["label"]] for r in dsets["dev"]]
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            preds.extend(
                model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
            )
    compwa = comp_weighted_accuracy(dev_seqs, y_true, preds)
    print(f"Depth {depth} | Complexity Weighted Accuracy (dev): {compwa:.4f}")
    ed["predictions"] = preds
    ed["ground_truth"] = y_true
    ed["compWA"] = compwa

    # ---- plot loss curve for this depth ----
    plt.figure()
    plt.plot(ed["losses"]["train"], label="train")
    plt.plot(ed["losses"]["val"], label="val")
    plt.title(f"Loss curve – {depth} GCN layers")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_depth_{depth}.png"))
    plt.close()

# ------------ save experiment data -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll experiments finished. Data saved to ./working/experiment_data.npy")
