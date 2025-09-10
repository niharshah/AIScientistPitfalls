import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from collections import defaultdict

#####################################################################
# working dir + device
#####################################################################
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#####################################################################
# load / fallback SPR_BENCH
#####################################################################
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for spl in ("train", "dev", "test"):
        d[spl] = _load(f"{spl}.csv")
    return d


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():
    print("SPR_BENCH not found – creating tiny synthetic data so the script can run.")
    os.makedirs(DATA_PATH, exist_ok=True)
    rng = np.random.default_rng(0)
    shapes = list("ABCDE")
    colours = list("12345")
    for split, n in (("train", 500), ("dev", 100), ("test", 100)):
        with open(DATA_PATH / f"{split}.csv", "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                length = rng.integers(4, 9)
                seq = " ".join(
                    rng.choice(shapes) + rng.choice(colours) for _ in range(length)
                )
                lbl = rng.choice(["yes", "no"])
                f.write(f"{split}_{i},{seq},{lbl}\n")

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


#####################################################################
# utility functions for metrics
#####################################################################
def count_color_variety(sequence: str) -> int:
    return len(set(t[1:] if len(t) > 1 else "0" for t in sequence.strip().split()))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def complexity_weight(sequence: str) -> int:
    return count_color_variety(sequence) * count_shape_variety(sequence)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    correct = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


#####################################################################
# build vocabularies
#####################################################################
def parse_token(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


shape_set, colour_set = set(), set()
for row in dsets["train"]:
    for tok in row["sequence"].split():
        s, c = parse_token(tok)
        shape_set.add(s)
        colour_set.add(c)
shape2id = {s: i for i, s in enumerate(sorted(shape_set))}
col2id = {c: i for i, c in enumerate(sorted(colour_set))}
label2id = {l: i for i, l in enumerate(sorted({r["label"] for r in dsets["train"]}))}
print("Shapes:", shape2id)
print("Colours:", col2id)
print("Labels:", label2id)

feat_dim = len(shape2id) + len(col2id)


#####################################################################
# sequence -> graph
#####################################################################
def seq_to_graph(sequence: str, label: str):
    toks = sequence.split()
    n = len(toks)
    # node features (one-hot shape | one-hot colour)
    x = torch.zeros((n, feat_dim), dtype=torch.float32)
    for i, tok in enumerate(toks):
        s, c = parse_token(tok)
        x[i, shape2id[s]] = 1.0
        x[i, len(shape2id) + col2id[c]] = 1.0
    # edges: consecutive + same-shape + same-color (bidirectional)
    src, dst = [], []
    # positional edges
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    # shared attributes
    shape_groups = defaultdict(list)
    colour_groups = defaultdict(list)
    for i, tok in enumerate(toks):
        s, c = parse_token(tok)
        shape_groups[s].append(i)
        colour_groups[c].append(i)
    for group in list(shape_groups.values()) + list(colour_groups.values()):
        for i in group:
            for j in group:
                if i != j:
                    src.append(i)
                    dst.append(j)
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.zeros((2, 0), dtype=torch.long)
    )
    y = torch.tensor([label2id[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train = build_graph_dataset("train")
graph_dev = build_graph_dataset("dev")
graph_test = build_graph_dataset("test")

train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)
test_loader = DataLoader(graph_test, batch_size=128, shuffle=False)


#####################################################################
# model
#####################################################################
class GraphModel(torch.nn.Module):
    def __init__(self, in_dim, hid=128, num_classes=len(label2id)):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.conv3 = SAGEConv(hid, hid)
        self.norm1 = torch.nn.LayerNorm(hid)
        self.norm2 = torch.nn.LayerNorm(hid)
        self.drop = torch.nn.Dropout(0.3)
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.drop(self.norm1(self.conv1(x, edge_index).relu()))
        x = self.drop(self.norm2(self.conv2(x, edge_index).relu()))
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = GraphModel(feat_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

#####################################################################
# experiment tracking dict
#####################################################################
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},  # will store dict per epoch
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

#####################################################################
# training loop with early stopping on CXA
#####################################################################
MAX_EPOCH = 30
patience = 5
best_cxa = -1.0
stagnant = 0


def evaluate(loader, seqs):
    model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.argmax(dim=-1).cpu().tolist())
            gts.extend(batch.y.cpu().tolist())
    avg_loss = total_loss / len(seqs)
    acc = np.mean(np.array(preds) == np.array(gts))
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cxa = complexity_weighted_accuracy(seqs, gts, preds)
    return avg_loss, acc, cwa, swa, cxa, preds


for epoch in range(1, MAX_EPOCH + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    train_loss = epoch_loss / len(graph_train)

    # quick train accuracy for logging
    tr_loss, tr_acc, tr_cwa, tr_swa, tr_cxa, _ = evaluate(
        train_loader, [r["sequence"] for r in dsets["train"]]
    )

    # validation
    val_loss, val_acc, val_cwa, val_swa, val_cxa, val_preds = evaluate(
        dev_loader, [r["sequence"] for r in dsets["dev"]]
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"acc": tr_acc, "cwa": tr_cwa, "swa": tr_swa, "cxa": tr_cxa}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "cwa": val_cwa, "swa": val_swa, "cxa": val_cxa}
    )

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CXA = {val_cxa:.4f}")

    # early stopping
    if val_cxa > best_cxa + 1e-4:
        best_cxa = val_cxa
        stagnant = 0
        # keep best predictions for later analysis
        experiment_data["SPR_BENCH"]["predictions"] = val_preds
        experiment_data["SPR_BENCH"]["ground_truth"] = [
            label2id[r["label"]] for r in dsets["dev"]
        ]
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        stagnant += 1
        if stagnant >= patience:
            print("Early stopping triggered.")
            break

#####################################################################
# plot loss & CXA
#####################################################################
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Cross-Entropy Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

plt.figure()
plt.plot(
    [d["cxa"] for d in experiment_data["SPR_BENCH"]["metrics"]["train"]],
    label="train_CXA",
)
plt.plot(
    [d["cxa"] for d in experiment_data["SPR_BENCH"]["metrics"]["val"]], label="val_CXA"
)
plt.title("Complexity-Weighted Accuracy")
plt.legend()
plt.savefig(os.path.join(working_dir, "cxa_curve.png"))

#####################################################################
# save all experiment data
#####################################################################
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All artefacts saved to ./working")
