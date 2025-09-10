import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- experiment dict ----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_cpxwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- helper : load SPR_BENCH ----------------
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
if not DATA_PATH.exists():  # build tiny synthetic substitute
    print("SPR_BENCH not found – synthesising tiny dataset for demo.")
    os.makedirs(DATA_PATH, exist_ok=True)
    for split, n_rows in [("train", 200), ("dev", 40), ("test", 40)]:
        rng = np.random.default_rng(0)
        shapes, colours = ["A", "B", "C"], ["1", "2", "3"]
        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            import csv

            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n_rows):
                tok_cnt = rng.integers(3, 7)
                seq = " ".join(
                    rng.choice(shapes) + rng.choice(colours) for _ in range(tok_cnt)
                )
                w.writerow([f"{split}_{i}", seq, rng.choice(["yes", "no"])])

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------------- vocab building ----------------
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
id2label = {v: k for k, v in label2id.items()}
in_dim = len(shape2id) + len(col2id)


# ---------------- sequence -> graph (self-loop edges) ----------------
def seq_to_graph(sequence, lbl):
    tokens = sequence.split()
    x_list = []
    for tok in tokens:
        s, c = parse_token(tok)
        vec = np.zeros(in_dim, dtype=np.float32)
        vec[shape2id[s]] = 1.0
        vec[len(shape2id) + col2id[c]] = 1.0
        x_list.append(vec)
    x = torch.tensor(np.stack(x_list), dtype=torch.float)
    # self-loop edges to enable message passing
    num_nodes = x.size(0)
    idx = torch.arange(num_nodes, dtype=torch.long)
    edge_index = torch.stack([idx, idx], dim=0)  # shape (2, num_nodes)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train = build_graph_dataset("train")
graph_dev = build_graph_dataset("dev")

train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)


# ---------------- model ----------------
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


model = GCN(in_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------- metrics ----------------
def complexity_weight(seq):
    toks = seq.split()
    shapes = {t[0] for t in toks}
    cols = {t[1:] if len(t) > 1 else "0" for t in toks}
    return len(shapes) + len(cols)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    hit = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(hit) / sum(w) if sum(w) else 0.0


# ---------------- training loop ----------------
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tr_loss = tr_corr = tr_ex = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * batch.num_graphs
        tr_corr += int((out.argmax(-1) == batch.y).sum().item())
        tr_ex += batch.num_graphs
    tr_loss /= tr_ex
    tr_acc = tr_corr / tr_ex

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
    v_loss /= v_ex
    v_acc = v_corr / v_ex
    seqs_dev = [r["sequence"] for r in dsets["dev"]]
    gt_dev = [label2id[r["label"]] for r in dsets["dev"]]
    cpxwa = complexity_weighted_accuracy(seqs_dev, gt_dev, preds)

    # ---- log ----
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(v_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_cpxwa"].append(cpxwa)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={tr_loss:.4f}  val_loss={v_loss:.4f} "
        f"val_acc={v_acc:.4f}  CpxWA={cpxwa:.4f}"
    )

# save final predictions & ground-truth
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gt_dev

# ---------------- plots & save ----------------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Cross-Entropy Loss – self-loop fix")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve_SPR_BENCH_selfloop.png"))

plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["metrics"]["val_cpxwa"])
plt.title("Complexity-Weighted Accuracy (dev)")
plt.savefig(os.path.join(working_dir, "cpxwa_curve_SPR_BENCH_selfloop.png"))

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All artifacts saved to ./working")
