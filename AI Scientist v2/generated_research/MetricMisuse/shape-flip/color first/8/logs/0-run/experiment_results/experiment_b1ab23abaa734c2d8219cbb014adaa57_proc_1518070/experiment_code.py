import os, pathlib, random, time, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ----------------- I/O & DEVICE -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- load / synthesize SPR_BENCH -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():  # tiny synthetic fallback
    print("SPR_BENCH not found – creating tiny synthetic data.")
    os.makedirs(DATA_PATH, exist_ok=True)
    rng = np.random.default_rng(0)
    shapes, colours = ["A", "B", "C"], ["1", "2", "3"]
    for split, n_ex in [("train", 200), ("dev", 40), ("test", 40)]:
        with open(DATA_PATH / f"{split}.csv", "w", newline="") as f:
            import csv

            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n_ex):
                toks = " ".join(
                    rng.choice(shapes) + rng.choice(colours)
                    for _ in range(rng.integers(3, 7))
                )
                w.writerow([f"{split}_{i}", toks, rng.choice(["yes", "no"])])
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ----------------- vocab -----------------
def parse_token(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


shapes, colours = set(), set()
for row in dsets["train"]:
    for t in row["sequence"].split():
        s, c = parse_token(t)
        shapes.add(s)
        colours.add(c)
shape2id = {s: i for i, s in enumerate(sorted(shapes))}
col2id = {c: i for i, c in enumerate(sorted(colours))}
label2id = {l: i for i, l in enumerate(sorted({r["label"] for r in dsets["train"]}))}
print("Shapes:", shape2id, "\nColours:", col2id)


# ----------------- helper: complexity weight -----------------
def complexity_weight(seq):
    toks = seq.split()
    return len({t[0] for t in toks}) + len({t[1:] if len(t) > 1 else "0" for t in toks})


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / max(sum(w), 1)


# ----------------- graph construction -----------------
def seq_to_graph(sequence, lbl, shuffle=False, rng=None):
    toks = sequence.split()
    if shuffle and len(toks) > 1:
        rng.shuffle(toks)
    n = len(toks)
    # node features
    x = np.zeros((n, len(shape2id) + len(col2id)), dtype=np.float32)
    for i, tok in enumerate(toks):
        s, c = parse_token(tok)
        x[i, shape2id[s]] = 1.0
        x[i, len(shape2id) + col2id[c]] = 1.0
    x = torch.tensor(x)
    # consecutive edges (undirected)
    if n > 1:
        src = torch.arange(n - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split, shuffle=False, seed=0):
    rng = random.Random(seed)
    return [seq_to_graph(r["sequence"], r["label"], shuffle, rng) for r in dsets[split]]


# ----------------- model -----------------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64, n_cls=len(label2id)):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hid), GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, n_cls)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ----------------- experiment runner -----------------
def run_experiment(
    ablation_name: str, shuffle_tokens: bool, seed: int = 0, epochs: int = 10
):
    print(f"\n=== Running {ablation_name} (shuffle={shuffle_tokens}) ===")
    # datasets / loaders
    graph_train = build_graph_dataset("train", shuffle_tokens, seed)
    graph_dev = build_graph_dataset("dev", shuffle_tokens, seed + 1)
    graph_test = build_graph_dataset("test", shuffle_tokens, seed + 2)
    train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
    dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)
    # model / opt
    torch.manual_seed(seed)
    model = GCN(in_dim=len(shape2id) + len(col2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # tracking
    track = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # training loop
    for ep in range(1, epochs + 1):
        model.train()
        tloss = tcorrect = nex = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            tloss += loss.item() * batch.num_graphs
            tcorrect += int((out.argmax(-1) == batch.y).sum().item())
            nex += batch.num_graphs
        track["losses"]["train"].append(tloss / nex)
        track["metrics"]["train"].append(tcorrect / nex)
        # ---- validation ----
        model.eval()
        vloss = vcorr = vex = 0
        preds = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                vloss += F.cross_entropy(out, batch.y).item() * batch.num_graphs
                p = out.argmax(-1).cpu()
                preds.extend(p.tolist())
                vcorr += int((p == batch.y.cpu()).sum().item())
                vex += batch.num_graphs
        track["losses"]["val"].append(vloss / vex)
        track["metrics"]["val"].append(vcorr / vex)
        print(f"Epoch {ep:02d}: val_loss={vloss/vex:.4f}, val_acc={vcorr/vex:.3f}")
    # complexity-weighted accuracy on dev
    dev_seqs = [r["sequence"] for r in dsets["dev"]]
    y_true = [label2id[r["label"]] for r in dsets["dev"]]
    y_pred = preds
    cwa = comp_weighted_accuracy(dev_seqs, y_true, y_pred)
    print(f"{ablation_name} Comp-WA (dev): {cwa:.4f}")
    track["predictions"], track["ground_truth"] = y_pred, y_true
    track["comp_weighted_accuracy"] = cwa
    # plot
    plt.figure()
    plt.plot(track["losses"]["train"], label="train")
    plt.plot(track["losses"]["val"], label="val")
    plt.title(f"Loss – {ablation_name}")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{ablation_name}.png"))
    plt.close()
    return track


# ----------------- run baseline and shuffled -----------------
experiment_data = {
    "baseline": {"SPR_BENCH": run_experiment("baseline", shuffle_tokens=False, seed=0)},
    "order_shuffled": {
        "SPR_BENCH": run_experiment("order_shuffled", shuffle_tokens=True, seed=42)
    },
}

# ----------------- save -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All data saved to", working_dir)
