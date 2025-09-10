import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- misc ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for s in ["train", "dev", "test"]:
        out[s] = _load(f"{s}.csv")
    return out


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():  # tiny synthetic fallback
    print("SPR_BENCH not found – creating tiny synthetic data.")
    DATA_PATH.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split, nrows in [("train", 200), ("dev", 40), ("test", 40)]:
        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            import csv

            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(nrows):
                n = rng.integers(3, 7)
                seq = " ".join(
                    rng.choice(list("ABC")) + rng.choice(list("123")) for _ in range(n)
                )
                w.writerow([f"{split}_{i}", seq, rng.choice(["yes", "no"])])
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
print("Shapes:", shape2id, "\nColours:", col2id, "\nLabels:", label2id)


# ---------- sequence ➜ graph ----------
def seq_to_graph(sequence, lbl):
    toks = sequence.split()
    feats = []
    for tok in toks:
        s, c = parse_token(tok)
        vec = np.zeros(len(shape2id) + len(col2id), np.float32)
        vec[shape2id[s]] = 1.0
        vec[len(shape2id) + col2id[c]] = 1.0
        feats.append(vec)
    x = torch.tensor(np.stack(feats))
    n = len(toks)
    if n > 1:
        src = torch.arange(n - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]])
    return Data(x=x, edge_index=edge_index, y=y)


def build_split(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train, graph_dev, graph_test = map(build_split, ["train", "dev", "test"])
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128)


# ---------- metrics ----------
def complexity_weight(seq):
    t = seq.split()
    return len({tok[0] for tok in t}) + len(
        {tok[1:] if len(tok) > 1 else "0" for tok in t}
    )


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) else 0.0


# ---------- model ----------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, n_classes, dropout):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


# ---------- experiment container ----------
experiment_data = {}
dropout_rates = [0.0, 0.1, 0.3, 0.5]
EPOCHS = 10

for dr in dropout_rates:
    tag = f"dropout_{dr}"
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = GCN(
        in_dim=len(shape2id) + len(col2id), hid=64, n_classes=len(label2id), dropout=dr
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- train epochs ----
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot_loss = tot_cor = tot_ex = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * batch.num_graphs
            tot_cor += int((out.argmax(-1) == batch.y).sum())
            tot_ex += batch.num_graphs
        tr_loss, tr_acc = tot_loss / tot_ex, tot_cor / tot_ex
        experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["train"].append(tr_acc)

        # ---- validation ----
        model.eval()
        v_loss = v_cor = v_ex = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                v_loss += F.cross_entropy(out, batch.y).item() * batch.num_graphs
                v_cor += int((out.argmax(-1) == batch.y).sum())
                v_ex += batch.num_graphs
        val_loss, val_acc = v_loss / v_ex, v_cor / v_ex
        experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(val_acc)
        print(f"[{tag}] Epoch {ep:02d}  val_loss={val_loss:.4f}")

    # ---- final dev evaluation for CompWA ----
    seqs = [r["sequence"] for r in dsets["dev"]]
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            preds.extend(
                model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
            )
    compwa = comp_weighted_accuracy(
        seqs, [label2id[r["label"]] for r in dsets["dev"]], preds
    )
    print(f"[{tag}] Complexity-Weighted Accuracy: {compwa:.4f}")
    edict = experiment_data[tag]["SPR_BENCH"]
    edict["predictions"] = preds
    edict["ground_truth"] = [label2id[r["label"]] for r in dsets["dev"]]
    edict["comp_weighted_acc"] = compwa

    # ---- plot ----
    plt.figure()
    plt.plot(edict["losses"]["train"], label="train")
    plt.plot(edict["losses"]["val"], label="val")
    plt.title(f"Loss curve – {tag}")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{tag}.png"))
    plt.close()

# ---------- save all ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All data saved to ./working/experiment_data.npy")
