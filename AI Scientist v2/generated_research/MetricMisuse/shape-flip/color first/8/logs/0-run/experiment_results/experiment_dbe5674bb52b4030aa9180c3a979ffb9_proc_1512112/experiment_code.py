import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ----- working dir -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----- helper: load SPR_BENCH -----
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
if not DATA_PATH.exists():  # tiny synthetic fallback
    print("SPR_BENCH not found – creating tiny synthetic data.")
    os.makedirs(DATA_PATH, exist_ok=True)
    for split, s in [("train", 200), ("dev", 40), ("test", 40)]:
        seqs, labels = [], []
        shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
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


# ----- vocab -----
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


# ----- graph conversion -----
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
        src = torch.arange(0, n - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]])
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train, graph_dev, graph_test = map(build_graph_dataset, ["train", "dev", "test"])

train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)


# ----- model -----
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


# ----- Complexity-Weighted Accuracy -----
def complexity_weight(seq):
    toks = seq.split()
    shapes = {t[0] for t in toks}
    cols = {t[1:] if len(t) > 1 else "0" for t in toks}
    return len(shapes) + len(cols)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


# ----- experiment logging dict -----
experiment_data = {"learning_rate": {}}

# ----- hyper-parameter grid -----
learning_rates = [5e-4, 1e-3, 2e-3, 5e-3]
EPOCHS = 10

for lr in learning_rates:
    print(f"\n=== Training with learning rate {lr} ===")
    model = GCN(in_dim=len(shape2id) + len(col2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
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
        tr_loss, tr_acc = tot_loss / tot_ex, tot_corr / tot_ex
        run_log["losses"]["train"].append(tr_loss)
        run_log["metrics"]["train"].append(tr_acc)

        # ---- validation ----
        model.eval()
        v_loss = v_corr = v_ex = 0
        preds = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                v_loss += loss.item() * batch.num_graphs
                pred = out.argmax(-1).cpu()
                preds.extend(pred.tolist())
                v_corr += int((pred == batch.y.cpu()).sum().item())
                v_ex += batch.num_graphs
        val_loss, val_acc = v_loss / v_ex, v_corr / v_ex
        run_log["losses"]["val"].append(val_loss)
        run_log["metrics"]["val"].append(val_acc)
        print(f"  Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

    # ----- dev CompWA -----
    seqs_dev = [r["sequence"] for r in dsets["dev"]]
    gts_dev = [label2id[r["label"]] for r in dsets["dev"]]
    compwa = comp_weighted_accuracy(seqs_dev, gts_dev, preds)
    run_log["metrics"]["comp_weighted_accuracy"] = compwa
    run_log["predictions"] = preds
    run_log["ground_truth"] = gts_dev
    print(f"  Complexity-Weighted Accuracy: {compwa:.4f}")

    # ----- store -----
    experiment_data["learning_rate"][str(lr)] = run_log

    # quick plot for this lr
    plt.figure()
    plt.plot(run_log["losses"]["train"], label="train")
    plt.plot(run_log["losses"]["val"], label="val")
    plt.title(f"LR {lr} – Cross-Entropy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_lr_{lr}.png"))
    plt.close()

# ----- save all experiment data -----
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll data & plots saved to ./working")
