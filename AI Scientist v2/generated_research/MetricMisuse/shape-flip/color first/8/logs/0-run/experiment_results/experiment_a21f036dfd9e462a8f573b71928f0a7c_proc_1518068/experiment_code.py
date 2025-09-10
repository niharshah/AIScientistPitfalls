# Depth-1 GCN Ablation – single-file, self-contained
import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ------------------- bookkeeping -------------------
ablation_name = "depth1_gcn"  # <- ablation identifier
dataset_name = "SPR_BENCH"

experiment_data = {
    ablation_name: {
        dataset_name: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------- env / dirs --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- dataset helpers ---------------
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
if not DATA_PATH.exists():  # fallback synthetic tiny data
    print("SPR_BENCH not found – building tiny synthetic data.")
    os.makedirs(DATA_PATH, exist_ok=True)
    rng = np.random.default_rng(0)
    shapes = ["A", "B", "C"]
    colours = ["1", "2", "3"]
    for split, sz in [("train", 200), ("dev", 40), ("test", 40)]:
        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            import csv

            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(sz):
                n = rng.integers(3, 7)
                seq = " ".join(
                    rng.choice(shapes) + rng.choice(colours) for _ in range(n)
                )
                label = rng.choice(["yes", "no"])
                w.writerow([f"{split}_{i}", seq, label])

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ------------------- vocab / label mapping ----------
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
print("Shapes:", shape2id, "\nColours:", col2id, "\nLabels:", label2id)


# ------------------- seq -> graph -------------------
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
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train = build_graph_dataset("train")
graph_dev = build_graph_dataset("dev")
graph_test = build_graph_dataset("test")

train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)
test_loader = DataLoader(graph_test, batch_size=128, shuffle=False)


# ------------------- Depth-1 GCN model --------------
class OneLayerGCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64, num_classes=len(label2id)):
        super().__init__()
        self.conv = GCNConv(in_dim, hid)
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = OneLayerGCN(len(shape2id) + len(col2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------- complexity-weighted acc --------
def complexity_weight(seq):
    toks = seq.split()
    return len({t[0] for t in toks}) + len({t[1:] if len(t) > 1 else "0" for t in toks})


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) > 0 else 0.0


# ------------------- training -----------------------
EPOCHS = 10
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
    tr_loss = tot_loss / tot_ex
    tr_acc = tot_corr / tot_ex
    experiment_data[ablation_name][dataset_name]["losses"]["train"].append(tr_loss)
    experiment_data[ablation_name][dataset_name]["metrics"]["train"].append(tr_acc)

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
    val_loss = v_loss / v_ex
    val_acc = v_corr / v_ex
    experiment_data[ablation_name][dataset_name]["losses"]["val"].append(val_loss)
    experiment_data[ablation_name][dataset_name]["metrics"]["val"].append(val_acc)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

# ------------------- final CompWA on dev ------------
seqs_dev = [r["sequence"] for r in dsets["dev"]]
gt_dev = [label2id[r["label"]] for r in dsets["dev"]]
compwa = comp_weighted_accuracy(seqs_dev, gt_dev, preds)
print(f"Complexity-Weighted Accuracy (dev): {compwa:.4f}")

experiment_data[ablation_name][dataset_name]["predictions"] = preds
experiment_data[ablation_name][dataset_name]["ground_truth"] = gt_dev

# ------------------- plot & save --------------------
plt.figure()
plt.plot(experiment_data[ablation_name][dataset_name]["losses"]["train"], label="train")
plt.plot(experiment_data[ablation_name][dataset_name]["losses"]["val"], label="val")
plt.legend()
plt.title("Cross-Entropy Loss")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Artifacts saved to ./working")
