import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from datetime import datetime
from typing import List

# ----------------------------- helpers ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [ww if t == p else 0 for ww, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# ------------------------ load or create dataset -----------------------------
spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
have_real = spr_root.exists()
if have_real:
    from datasets import load_dataset

    def _load(csv_name):  # small util
        return load_dataset(
            "csv",
            data_files=str(spr_root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dsets = {split: _load(f"{split}.csv") for split in ["train", "dev", "test"]}
    print("Loaded real SPR_BENCH.")
else:
    print("SPR_BENCH not found, using synthetic.")
    rng = np.random.default_rng(0)

    def synth(n):
        shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
        seqs, labs = [], []
        for _ in range(n):
            length = rng.integers(4, 8)
            seq = " ".join(
                rng.choice(shapes) + rng.choice(colors) for _ in range(length)
            )
            seqs.append(seq)
            labs.append(rng.integers(0, 3))
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    dsets = {"train": synth(500), "dev": synth(100), "test": synth(100)}

# ---------------------------- vocabularies -----------------------------------
all_shapes, all_colors, all_labels = set(), set(), set()
for s in dsets["train"]["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0])
        all_colors.add(tok[1])
for l in dsets["train"]["label"]:
    all_labels.add(l)
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
num_shapes, num_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)

# --------------------------- graph builder -----------------------------------
from torch_geometric.data import Data


def seq_to_graph(seq: str, label: int):
    toks = seq.strip().split()
    n = len(toks)
    s_ids = [shape2idx[t[0]] for t in toks]
    c_ids = [color2idx[t[1]] for t in toks]
    x = torch.tensor(np.stack([s_ids, c_ids], 1), dtype=torch.long)
    if n > 1:
        src = np.arange(n - 1)
        dst = np.arange(1, n)
        edge_index = torch.tensor(
            np.vstack([np.hstack([src, dst]), np.hstack([dst, src])]), dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def build(split):
    if isinstance(split, dict):
        return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]
    return [seq_to_graph(rec["sequence"], rec["label"]) for rec in split]


train_data, dev_data, test_data = map(
    build, (dsets["train"], dsets["dev"], dsets["test"])
)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


# ------------------------------- model ---------------------------------------
class SPRGNN(nn.Module):
    def __init__(self, n_shapes, n_colors, n_cls):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shapes, 8)
        self.color_emb = nn.Embedding(n_colors, 8)
        self.lin = nn.Linear(16, 32)
        self.conv1, self.conv2 = GraphConv(32, 64), GraphConv(64, 64)
        self.cls = nn.Linear(64, n_cls)

    def forward(self, data):
        e1 = self.shape_emb(data.x[:, 0])
        e2 = self.color_emb(data.x[:, 1])
        x = F.relu(self.lin(torch.cat([e1, e2], 1)))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.cls(x)


# ------------------------ hyper-parameter sweep ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
weight_decays = [0, 1e-5, 1e-4, 5e-4, 1e-3]
experiment_data = {"weight_decay": {}}
epochs = 5
criterion = nn.CrossEntropyLoss()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for wd in weight_decays:
    tag = f"wd_{wd}"
    print(f"\n===== Training with weight_decay={wd} =====")
    exp = {
        "metrics": {"train_compwa": [], "val_compwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    model = SPRGNN(num_shapes, num_colors, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs
        avg_train_loss = total / len(train_loader.dataset)
        # ---- val ----
        model.eval()
        vtot = 0
        seqs, true, pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                vtot += loss.item() * batch.num_graphs
                preds = out.argmax(1).cpu().tolist()
                seqs.extend(batch.seq)
                true.extend(batch.y.cpu().tolist())
                pred.extend(preds)
        avg_val_loss = vtot / len(dev_loader.dataset)
        compwa = complexity_weighted_accuracy(seqs, true, pred)

        exp["losses"]["train"].append(avg_train_loss)
        exp["losses"]["val"].append(avg_val_loss)
        exp["metrics"]["val_compwa"].append(compwa)
        exp["epochs"].append(epoch)
        print(
            f"Epoch {epoch}: train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f} | CompWA {compwa:.4f}"
        )

    # ------------- test evaluation ------------------------------------------
    model.eval()
    seqs, true, pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out.argmax(1).cpu().tolist()
            seqs.extend(batch.seq)
            true.extend(batch.y.cpu().tolist())
            pred.extend(preds)
    test_compwa = complexity_weighted_accuracy(seqs, true, pred)
    exp["metrics"]["test_compwa"] = test_compwa
    exp["predictions"] = pred
    exp["ground_truth"] = true
    print(f"Test CompWA for wd={wd}: {test_compwa:.4f}")

    experiment_data["weight_decay"][tag] = exp

# ---------------------------- save and plot ----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
# simple plot summarising CompWA per wd
plt.figure()
vals = [
    experiment_data["weight_decay"][f"wd_{wd}"]["metrics"]["test_compwa"]
    for wd in weight_decays
]
plt.plot(weight_decays, vals, marker="o")
plt.xscale("log")
plt.xlabel("Weight decay")
plt.ylabel("Test CompWA")
plt.title("Weight-decay sweep")
plt.savefig(os.path.join(working_dir, f"wd_sweep_{timestamp}.png"))
print("All done, results stored in ./working")
