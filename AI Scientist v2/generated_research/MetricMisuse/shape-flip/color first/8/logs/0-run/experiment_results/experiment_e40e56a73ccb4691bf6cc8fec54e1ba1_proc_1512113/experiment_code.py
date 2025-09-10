import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ------------ working dir ------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ device ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------ load SPR_BENCH ------------
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
    for split, s in [("train", 200), ("dev", 40), ("test", 40)]:
        seqs, labels, shapes, colors = [], [], ["A", "B", "C"], ["1", "2", "3"]
        rng = np.random.default_rng(0)
        for _ in range(s):
            n = rng.integers(3, 7)
            seqs.append(
                " ".join(rng.choice(shapes) + rng.choice(colors) for _ in range(n))
            )
            labels.append(rng.choice(["yes", "no"]))
        import csv

        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i, (seq, lbl) in enumerate(zip(seqs, labels)):
                w.writerow([f"{split}_{i}", seq, lbl])

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ------------ vocab / label maps ------------
def parse_token(tok):  # token like 'A1' or 'B'
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
print("Shapes:", shape2id)
print("Colours:", col2id)


# ------------ seq -> graph ------------
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
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train, graph_dev, graph_test = map(build_graph_dataset, ["train", "dev", "test"])


# ------------ model ------------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64, num_classes=len(label2id)):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hid), GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ------------ helper metrics ------------
def complexity_weight(seq):
    toks = seq.split()
    return len({t[0] for t in toks}) + len({t[1:] if len(t) > 1 else "0" for t in toks})


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) else 0.0


# ------------ experiment data dict ------------
experiment_data = {"batch_size": {}}


# ------------ training function ------------
def run_experiment(batch_size, epochs=10):
    tag = f"batch_{batch_size}"
    experiment_data["batch_size"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    train_loader = DataLoader(graph_train, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(
        graph_dev, batch_size=128, shuffle=False
    )  # eval with large batch
    model = GCN(in_dim=len(shape2id) + len(col2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        # ---- train ----
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
        experiment_data["batch_size"][tag]["losses"]["train"].append(tot_loss / tot_ex)
        experiment_data["batch_size"][tag]["metrics"]["train"].append(tot_corr / tot_ex)

        # ---- validation ----
        model.eval()
        v_loss = v_corr = v_ex = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                v_loss += F.cross_entropy(out, batch.y).item() * batch.num_graphs
                v_corr += int((out.argmax(-1) == batch.y).sum().item())
                v_ex += batch.num_graphs
        experiment_data["batch_size"][tag]["losses"]["val"].append(v_loss / v_ex)
        experiment_data["batch_size"][tag]["metrics"]["val"].append(v_corr / v_ex)
        print(
            f"[{tag}] Epoch {epoch}: val_loss={v_loss / v_ex:.4f}, val_acc={v_corr / v_ex:.4f}"
        )

    # ---- final predictions & CompWA ----
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            preds.extend(
                model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
            )
    gts = [label2id[r["label"]] for r in dsets["dev"]]
    experiment_data["batch_size"][tag]["predictions"] = preds
    experiment_data["batch_size"][tag]["ground_truth"] = gts
    compwa = comp_weighted_accuracy([r["sequence"] for r in dsets["dev"]], gts, preds)
    print(f"[{tag}] Complexity-Weighted Accuracy (dev): {compwa:.4f}")


# ------------ run all batch sizes ------------
for bs in [16, 32, 64, 128]:
    run_experiment(bs)

# ------------ plot loss curves ------------
plt.figure()
for tag, dat in experiment_data["batch_size"].items():
    plt.plot(dat["losses"]["val"], label=f"{tag}")
plt.title("Validation Cross-Entropy vs Epoch")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

# ------------ save data ------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All data saved to ./working")
