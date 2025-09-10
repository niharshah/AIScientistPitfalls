import os, random, string, pathlib, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import List, Tuple
from datasets import load_dataset, DatasetDict

# -------------------- set up working dir & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- metric helpers --------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1e-6, sum(weights))


# -------------------- dataset loading / fallback --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


def generate_synthetic_dataset(n: int) -> Tuple[List[str], List[int]]:
    shapes = list(string.ascii_uppercase[:5])
    colors = list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seqs.append(
            " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        )
        labels.append(random.randint(0, 2))
    return seqs, labels


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    spr = load_spr_bench(data_root)
    print("Loaded real SPR_BENCH data")
except Exception as e:
    print("Falling back to synthetic data:", e)
    train_seq, train_y = generate_synthetic_dataset(500)
    dev_seq, dev_y = generate_synthetic_dataset(120)
    test_seq, test_y = generate_synthetic_dataset(120)
    spr = DatasetDict(
        {
            "train": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
            "dev": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
            "test": load_dataset(
                "json", data_files={"train": [{}]}, split="train"
            ).remove_columns([]),
        }
    )
    spr["train"] = (
        spr["train"].add_column("sequence", train_seq).add_column("label", train_y)
    )
    spr["dev"] = spr["dev"].add_column("sequence", dev_seq).add_column("label", dev_y)
    spr["test"] = (
        spr["test"].add_column("sequence", test_seq).add_column("label", test_y)
    )


# -------------------- vocab creation --------------------
def build_vocabs(dataset) -> Tuple[dict, dict, dict]:
    shapes, colors, labels = set(), set(), set()
    for ex in dataset:
        for tok in ex["sequence"].split():
            shapes.add(tok[0])
            colors.add(tok[1:])
        labels.add(ex["label"])
    return (
        {s: i for i, s in enumerate(sorted(shapes))},
        {c: i for i, c in enumerate(sorted(colors))},
        {l: i for i, l in enumerate(sorted(labels))},
    )


shape2idx, color2idx, label2idx = build_vocabs(spr["train"])
num_shapes, num_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)


# -------------------- graph conversion --------------------
def sequence_to_graph(seq: str, label: int) -> Data:
    tokens = seq.split()
    shape_idx = [shape2idx[t[0]] for t in tokens]
    color_idx = [color2idx[t[1:]] for t in tokens]
    x = torch.tensor(list(zip(shape_idx, color_idx)), dtype=torch.long)
    src, dst = [], []
    for i in range(len(tokens) - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


train_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["train"]]
dev_graphs = [sequence_to_graph(ex["sequence"], ex["label"]) for ex in spr["dev"]]


# -------------------- model --------------------
class SPRGraphNet(nn.Module):
    def __init__(self, num_shapes, num_colors, num_classes, emb_dim=16, hidden=32):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, emb_dim)
        self.color_emb = nn.Embedding(num_colors, emb_dim)
        self.gnn1 = SAGEConv(emb_dim * 2, hidden)
        self.gnn2 = SAGEConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, data):
        shp = self.shape_emb(data.x[:, 0])
        col = self.color_emb(data.x[:, 1])
        h = torch.cat([shp, col], dim=-1)
        h = self.gnn1(h, data.edge_index).relu()
        h = self.gnn2(h, data.edge_index).relu()
        hg = global_mean_pool(h, data.batch)
        return self.classifier(hg)


# -------------------- loaders --------------------
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=64)

# -------------------- experiment tracker --------------------
experiment_data = {"hidden_dim": {"SPR_BENCH": {}}}

hidden_dims = [16, 32, 64, 128]
epochs = 5
for hdim in hidden_dims:
    run_dict = {
        "metrics": {"train_cwa2": [], "val_cwa2": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    print(f"\n--- Training with hidden_dim = {hdim} ---")
    model = SPRGraphNet(num_shapes, num_colors, num_classes, hidden=hdim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)
        run_dict["losses"]["train"].append(avg_train_loss)

        # evaluate
        model.eval()
        val_loss = 0
        ys = []
        preds = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=-1).cpu().tolist()
                truth = batch.y.cpu().tolist()
                seqs.extend(batch.seq)
                ys.extend(truth)
                preds.extend(pred)
        avg_val_loss = val_loss / len(dev_loader.dataset)
        run_dict["losses"]["val"].append(avg_val_loss)
        cwa2_val = complexity_weighted_accuracy(seqs, ys, preds)
        run_dict["metrics"]["val_cwa2"].append(cwa2_val)

        print(
            f" hd={hdim} epoch={epoch}  val_loss={avg_val_loss:.4f}  CWA2={cwa2_val:.4f}"
        )

    run_dict["predictions"] = preds
    run_dict["ground_truth"] = ys
    experiment_data["hidden_dim"]["SPR_BENCH"][str(hdim)] = run_dict

# -------------------- save results --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
