import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from typing import List
from datetime import datetime


# ---------------------------------------------------------------------------#
#                           Helper functions                                  #
# ---------------------------------------------------------------------------#
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weighted_accuracy(
    sequences: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_color_variety(s) + count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    denom = sum(weights)
    return sum(correct) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------#
#                           Data preparation                                  #
# ---------------------------------------------------------------------------#
spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
have_real_data = spr_root.exists()
if have_real_data:
    from datasets import load_dataset

    def _load(csv_name):  # helper for CSV loading
        return load_dataset(
            "csv",
            data_files=str(spr_root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dsets = {sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]}
else:  # tiny synthetic toy data

    def make_synth(n):
        seqs, labels = [], []
        shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
        for _ in range(n):
            ln = np.random.randint(4, 8)
            seqs.append(
                " ".join(
                    np.random.choice(shapes) + np.random.choice(colors)
                    for _ in range(ln)
                )
            )
            labels.append(np.random.randint(0, 3))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    dsets = {"train": make_synth(500), "dev": make_synth(100), "test": make_synth(100)}

# build vocabularies
all_shapes, all_colors, all_labels = set(), set(), set()
for s in dsets["train"]["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0]), all_colors.add(tok[1])
for l in dsets["train"]["label"]:
    all_labels.add(l)
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
num_shapes, num_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)


def seq_to_graph(seq: str, label: int):
    toks = seq.strip().split()
    n = len(toks)
    shape_ids = [shape2idx[t[0]] for t in toks]
    color_ids = [color2idx[t[1]] for t in toks]
    x = torch.tensor(np.stack([shape_ids, color_ids], 1), dtype=torch.long)
    if n > 1:
        src, dst = np.arange(n - 1), np.arange(1, n)
        edge_index = torch.tensor(
            np.vstack([np.hstack([src, dst]), np.hstack([dst, src])]), dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def build_dataset(split):
    if isinstance(split, dict):
        return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]
    return [seq_to_graph(rec["sequence"], rec["label"]) for rec in split]


train_data, dev_data, test_data = (
    build_dataset(dsets[k]) for k in ["train", "dev", "test"]
)


# ---------------------------------------------------------------------------#
#                               Model                                         #
# ---------------------------------------------------------------------------#
class SPRGNN(nn.Module):
    def __init__(self, n_shapes, n_colors, n_cls):
        super().__init__()
        self.shape_emb, self.color_emb = nn.Embedding(n_shapes, 8), nn.Embedding(
            n_colors, 8
        )
        self.lin_node = nn.Linear(16, 32)
        self.conv1, self.conv2 = GraphConv(32, 64), GraphConv(64, 64)
        self.classifier = nn.Linear(64, n_cls)

    def forward(self, data):
        x = torch.cat([self.shape_emb(data.x[:, 0]), self.color_emb(data.x[:, 1])], 1)
        x = F.relu(self.lin_node(x))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)


# ---------------------------------------------------------------------------#
#                           Hyper-parameter sweep                             #
# ---------------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
epochs = 5
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"learning_rate": {"SPR_BENCH": {}}}

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()

for lr in lrs:
    print(f"\n===> Training with learning rate {lr:.0e}")
    model = SPRGNN(num_shapes, num_colors, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_key = f"{lr:.0e}"
    exp = {
        "metrics": {"val_compwa": [], "test_compwa": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    for epoch in range(1, epochs + 1):
        # training
        model.train()
        tr_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch.num_graphs
        tr_loss /= len(train_loader.dataset)
        # validation
        model.eval()
        val_loss, seqs, y_true, y_pred = 0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds = out.argmax(1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(batch.y.cpu().tolist())
                seqs.extend(batch.seq)
        val_loss /= len(dev_loader.dataset)
        compwa = complexity_weighted_accuracy(seqs, y_true, y_pred)
        # log
        exp["losses"]["train"].append(tr_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["val_compwa"].append(compwa)
        exp["epochs"].append(epoch)
        print(
            f"  Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} CompWA={compwa:.4f}"
        )

    # final test
    model.eval()
    seqs, y_true, y_pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch.y.cpu().tolist())
            seqs.extend(batch.seq)
    test_compwa = complexity_weighted_accuracy(seqs, y_true, y_pred)
    exp["metrics"]["test_compwa"] = test_compwa
    exp["predictions"], exp["ground_truth"] = y_pred, y_true
    print(f"  Test CompWA with lr {lr:.0e}: {test_compwa:.4f}")

    # save per-lr plots
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure()
    plt.plot(exp["epochs"], exp["losses"]["train"], label="train")
    plt.plot(exp["epochs"], exp["losses"]["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss (lr={lr:.0e})")
    plt.savefig(os.path.join(working_dir, f"loss_lr{lr_key}_{ts}.png"))
    plt.close()

    plt.figure()
    plt.plot(exp["epochs"], exp["metrics"]["val_compwa"], label="Val CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.legend()
    plt.title(f"Val CompWA (lr={lr:.0e})")
    plt.savefig(os.path.join(working_dir, f"compwa_lr{lr_key}_{ts}.png"))
    plt.close()

    experiment_data["learning_rate"]["SPR_BENCH"][lr_key] = exp

# save raw data
np.save("experiment_data.npy", experiment_data)
print(
    "All experiments complete. Raw data saved to experiment_data.npy and plots in ./working"
)
