# num_epochs hyperparameter tuning – single-file runnable script
import os, copy, pathlib, numpy as np, torch, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool

# ---------- Reproducibility --------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------- Working dir ------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- Helper metric ----------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weighted_accuracy(
    sequences: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    w = [count_color_variety(s) + count_shape_variety(s) for s in sequences]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if w else 0.0


# ---------- Load SPR_BENCH or synthetic --------------------------------------
def load_real_spr(root: pathlib.Path):
    from datasets import load_dataset

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]}


spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
if spr_root.exists():
    dsets = load_real_spr(spr_root)
    print("Loaded real SPR_BENCH.")
else:
    print("SPR_BENCH not found, using synthetic toy data.")

    def make_synth(n):
        shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
        seqs, labels = [], []
        for _ in range(n):
            length = np.random.randint(4, 8)
            seqs.append(
                " ".join(
                    np.random.choice(shapes) + np.random.choice(colors)
                    for _ in range(length)
                )
            )
            labels.append(np.random.randint(0, 3))
        return {"sequence": seqs, "label": labels}

    dsets = {"train": make_synth(500), "dev": make_synth(100), "test": make_synth(100)}

# ---------- Build vocab ------------------------------------------------------
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


def seq_to_graph(seq: str, label: int):
    toks = seq.strip().split()
    n = len(toks)
    shape_ids = [shape2idx[t[0]] for t in toks]
    color_ids = [color2idx[t[1]] for t in toks]
    x = torch.tensor(np.stack([shape_ids, color_ids], 1), dtype=torch.long)
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


def build_dataset(split):
    if isinstance(split, dict):
        return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]
    return [seq_to_graph(r["sequence"], r["label"]) for r in split]


train_data, dev_data, test_data = map(
    build_dataset, (dsets["train"], dsets["dev"], dsets["test"])
)


# ---------- Model definition -------------------------------------------------
class SPRGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, 8)
        self.color_emb = nn.Embedding(num_colors, 8)
        self.lin_node = nn.Linear(16, 32)
        self.conv1 = GraphConv(32, 64)
        self.conv2 = GraphConv(64, 64)
        self.cls = nn.Linear(64, num_classes)

    def forward(self, data):
        s = self.shape_emb(data.x[:, 0])
        c = self.color_emb(data.x[:, 1])
        x = F.relu(self.lin_node(torch.cat([s, c], 1)))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.cls(x)


# ---------- Dataloaders (re-used) -------------------------------------------
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# ---------- Experiment data dict --------------------------------------------
experiment_data = {
    "num_epochs_tuning": {
        "SPR_BENCH": {
            "hparam_values": [],
            "epochs_run": [],
            "metrics": {"train_compwa": [], "val_compwa": [], "test_compwa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- Training routine -------------------------------------------------
criterion = nn.CrossEntropyLoss()


def train_model(num_epochs: int, patience: int = 5):
    model = SPRGNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_val_loss = None, float("inf")
    patience_ctr = 0
    tr_losses, val_losses, val_compwas = [], [], []
    for ep in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        tot = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            optim.step()
            tot += loss.item() * batch.num_graphs
        tr_losses.append(tot / len(train_loader.dataset))
        # ---- val ----
        model.eval()
        vtot = 0
        seqs = true = pred = []
        seqs, true, pred = [], [], []
        with torch.no_grad():
            for b in dev_loader:
                b = b.to(device)
                out = model(b)
                loss = criterion(out, b.y)
                vtot += loss.item() * b.num_graphs
                seqs.extend(b.seq)
                true.extend(b.y.cpu().tolist())
                pred.extend(out.argmax(1).cpu().tolist())
        v_loss = vtot / len(dev_loader.dataset)
        val_losses.append(v_loss)
        compwa = complexity_weighted_accuracy(seqs, true, pred)
        val_compwas.append(compwa)
        # ---- early stopping ----
        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            print(f"Early stopping at epoch {ep}")
            break
    # load best model
    model.load_state_dict(best_state)
    return model, tr_losses, val_losses, val_compwas


# ---------- Hyperparameter grid ---------------------------------------------
epoch_grid = [5, 15, 30, 50]
best_idx, best_val = None, -1

for epochs in epoch_grid:
    print(f"\n=== Training with num_epochs={epochs} ===")
    model, tr_l, val_l, val_c = train_model(epochs)
    # final compwa on train (using last epoch metrics)
    train_comp = val_c[0] if val_c else 0.0  # placeholder not used
    experiment_data["num_epochs_tuning"]["SPR_BENCH"]["hparam_values"].append(epochs)
    experiment_data["num_epochs_tuning"]["SPR_BENCH"]["epochs_run"].append(len(tr_l))
    experiment_data["num_epochs_tuning"]["SPR_BENCH"]["losses"]["train"].append(tr_l)
    experiment_data["num_epochs_tuning"]["SPR_BENCH"]["losses"]["val"].append(val_l)
    experiment_data["num_epochs_tuning"]["SPR_BENCH"]["metrics"]["val_compwa"].append(
        val_c[-1] if val_c else 0.0
    )
    # ---- test evaluation ----
    model.eval()
    seqs, true, preds = [], [], []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            out = model(b)
            seqs.extend(b.seq)
            true.extend(b.y.cpu().tolist())
            preds.extend(out.argmax(1).cpu().tolist())
    test_cwa = complexity_weighted_accuracy(seqs, true, preds)
    experiment_data["num_epochs_tuning"]["SPR_BENCH"]["metrics"]["test_compwa"].append(
        test_cwa
    )
    experiment_data["num_epochs_tuning"]["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["num_epochs_tuning"]["SPR_BENCH"][
        "ground_truth"
    ] = true  # same for all
    print(
        f"Finished num_epochs={epochs} | Val CompWA={val_c[-1]:.4f} | Test CompWA={test_cwa:.4f}"
    )
    if val_c[-1] > best_val:
        best_val = val_c[-1]
        best_idx = len(epoch_grid) - len(epoch_grid[epoch_grid.index(epochs) :])

# ---------- Plot for best configuration -------------------------------------
best_epochs = experiment_data["num_epochs_tuning"]["SPR_BENCH"]["hparam_values"][
    best_idx
]
best_tr = experiment_data["num_epochs_tuning"]["SPR_BENCH"]["losses"]["train"][best_idx]
best_val = experiment_data["num_epochs_tuning"]["SPR_BENCH"]["losses"]["val"][best_idx]
best_cwa_curve = experiment_data["num_epochs_tuning"]["SPR_BENCH"]["metrics"][
    "val_compwa"
][best_idx]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure()
plt.plot(range(1, len(best_tr) + 1), best_tr, label="train")
plt.plot(range(1, len(best_val) + 1), best_val, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss (best num_epochs={best_epochs})")
plt.legend()
plt.savefig(os.path.join(working_dir, f"loss_best_{ts}.png"))
# single value compwa, plot as point
plt.figure()
plt.plot(len(best_val), best_cwa_curve, "ro")
plt.xlabel("Epoch")
plt.ylabel("Val CompWA")
plt.title("Best validation CompWA")
plt.savefig(os.path.join(working_dir, f"val_compwa_best_{ts}.png"))

# ---------- Save experiment data --------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All done. Data saved to", working_dir)
