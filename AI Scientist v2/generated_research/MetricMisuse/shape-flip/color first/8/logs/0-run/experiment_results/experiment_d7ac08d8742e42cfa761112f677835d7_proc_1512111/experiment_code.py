import os, pathlib, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- helper: load SPR_BENCH ----------
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
    for split, size in [("train", 200), ("dev", 40), ("test", 40)]:
        rng = np.random.default_rng(0)
        seqs, labels = [], []
        shapes = ["A", "B", "C"]
        colors = ["1", "2", "3"]
        for _ in range(size):
            n = rng.integers(3, 7)
            seqs.append(
                " ".join(rng.choice(shapes) + rng.choice(colors) for _ in range(n))
            )
            labels.append(rng.choice(["yes", "no"]))
        import csv

        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i, (s, l) in enumerate(zip(seqs, labels)):
                w.writerow([f"{split}_{i}", s, l])

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------- vocab ----------
def parse_token(tok):  # returns (shape, colour)
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


# ---------- seq -> graph ----------
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
        src = torch.arange(n - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0)
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


# ---------- metrics ----------
def complexity_weight(seq):
    toks = seq.split()
    return len({t[0] for t in toks}) + len(
        {(t[1:] if len(t) > 1 else "0") for t in toks}
    )


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) else 0.0


# ---------- experiment storage ----------
experiment_data = {"epochs_tuning": {}}  # top level = hyperparam type

# ---------- hyperparameter grid ----------
EPOCH_OPTIONS = [10, 20, 40, 60]
patience = 5

for max_epochs in EPOCH_OPTIONS:
    run_key = f"EPOCHS_{max_epochs}"
    experiment_data["epochs_tuning"][run_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = GCN(in_dim=len(shape2id) + len(col2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
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

        # ---- validation ----
        model.eval()
        v_loss = v_corr = v_ex = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                v_loss += loss.item() * batch.num_graphs
                v_corr += int((out.argmax(-1) == batch.y).sum().item())
                v_ex += batch.num_graphs
        val_loss = v_loss / v_ex
        val_acc = v_corr / v_ex

        # ---- store ----
        ed = experiment_data["epochs_tuning"][run_key]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(tr_acc)
        ed["metrics"]["val"].append(val_acc)
        print(f"[{run_key}] Epoch {epoch}/{max_epochs}  val_loss={val_loss:.4f}")

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss, best_state, wait = val_loss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ---- load best model & evaluate ----
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            preds.extend(
                model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
            )
    ground_truth = [label2id[r["label"]] for r in dsets["dev"]]
    experiment_data["epochs_tuning"][run_key]["predictions"] = preds
    experiment_data["epochs_tuning"][run_key]["ground_truth"] = ground_truth

    # ---- plot ----
    plt.figure()
    plt.plot(ed["losses"]["train"], label="train")
    plt.plot(ed["losses"]["val"], label="val")
    plt.title(f"Loss curve ({run_key})")
    plt.xlabel("epoch")
    plt.ylabel("CE loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{run_key}.png"))
    plt.close()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All data saved to", working_dir)
