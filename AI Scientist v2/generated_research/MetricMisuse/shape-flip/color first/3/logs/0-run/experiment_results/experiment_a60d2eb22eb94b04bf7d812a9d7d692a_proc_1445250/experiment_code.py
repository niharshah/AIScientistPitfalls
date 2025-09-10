import os, pathlib, time, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import List
from datasets import DatasetDict, load_dataset

# ------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# locate SPR_BENCH
def locate_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in candidates:
        if p and (p / "train.csv").exists():
            print(f"Found SPR_BENCH at: {p.resolve()}")
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


# ------------------------------------------------------------
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)


# ------------------------------------------------------------
def extract_tokens(seq: str) -> List[str]:
    return seq.strip().split()


token_set, label_set = set(), set()
for ex in spr["train"]:
    token_set.update(extract_tokens(ex["sequence"]))
    label_set.add(ex["label"])
token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
label2idx = {lab: i for i, lab in enumerate(sorted(label_set))}
idx2label = {i: lab for lab, i in label2idx.items()}


# ------------------------------------------------------------
def seq_to_data(example):
    seq = example["sequence"]
    tokens = extract_tokens(seq)
    node_idx = [token2idx.get(tok, 0) for tok in tokens]
    x = torch.tensor(node_idx, dtype=torch.long).unsqueeze(-1)
    if len(tokens) > 1:
        src = torch.arange(0, len(tokens) - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[example["label"]]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.seq = seq
    return data


train_graphs = [seq_to_data(ex) for ex in spr["train"]]
dev_graphs = [seq_to_data(ex) for ex in spr["dev"]]
test_graphs = [seq_to_data(ex) for ex in spr["test"]]

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ------------------------------------------------------------
class SPR_GCN(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.gcn1 = GCNConv(embed_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.embed(x.squeeze(-1))
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.drop(x)
        return self.lin(x)


# ------------------------------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss_sum += loss.item() * batch.num_graphs
            preds = out.argmax(-1).cpu().tolist()
            labs = batch.y.view(-1).cpu().tolist()
            all_preds += preds
            all_labels += labs
            all_seqs += batch.seq
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    bwa = (cwa + swa) / 2.0
    return loss_sum / len(loader.dataset), bwa, cwa, swa, all_preds, all_labels


# ------------------------------------------------------------
hidden_dim_grid = [32, 64, 128, 256]
num_epochs = 5

experiment_data = {
    "gcn_hidden_dim": {
        "SPR_BENCH": {
            "hidden_dims": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

best_bwa, best_dim, best_model_state = -1, None, None
history_best_curve = {"train": [], "val": []}

for hd in hidden_dim_grid:
    print(f"\n=== Training with hidden_dim={hd} ===")
    model = SPR_GCN(len(token2idx), 32, hd, len(label2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_curve, val_curve = [], []
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, val_bwa, val_cwa, val_swa, *_ = evaluate(model, dev_loader, criterion)
        train_bwa, *_ = evaluate(model, train_loader, criterion)[1:]

        train_curve.append(train_bwa)
        val_curve.append(val_bwa)

        print(
            f"Epoch {epoch}/{num_epochs} - train_loss {train_loss:.4f} "
            f"val_loss {val_loss:.4f}  val_BWA {val_bwa:.4f}"
        )

    # record
    exp = experiment_data["gcn_hidden_dim"]["SPR_BENCH"]
    exp["hidden_dims"].append(hd)
    exp["metrics"]["train"].append(train_curve[-1])
    exp["metrics"]["val"].append(val_curve[-1])
    exp["losses"]["train"].append(train_loss)
    exp["losses"]["val"].append(val_loss)
    exp["timestamps"].append(time.time())

    # keep best
    if val_curve[-1] > best_bwa:
        best_bwa, best_dim = val_curve[-1], hd
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        history_best_curve["train"] = train_curve
        history_best_curve["val"] = val_curve

# ------------------------------------------------------------
print(f"\nBest hidden_dim = {best_dim} with dev BWA = {best_bwa:.4f}")
best_model = SPR_GCN(len(token2idx), 32, best_dim, len(label2idx)).to(device)
best_model.load_state_dict(best_model_state)

criterion = nn.CrossEntropyLoss()
test_loss, test_bwa, test_cwa, test_swa, test_preds, test_labels = evaluate(
    best_model, test_loader, criterion
)
print(
    f"TEST -> loss {test_loss:.4f}  BWA {test_bwa:.4f}  CWA {test_cwa:.4f}  SWA {test_swa:.4f}"
)

# store predictions / ground truth for best
exp = experiment_data["gcn_hidden_dim"]["SPR_BENCH"]
exp["predictions"] = test_preds
exp["ground_truth"] = test_labels

# ------------------------------------------------------------
# Save data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ------------------------------------------------------------
# Plot best curve
epochs = np.arange(1, num_epochs + 1)
plt.figure()
plt.plot(epochs, history_best_curve["train"], label="Train BWA")
plt.plot(epochs, history_best_curve["val"], label="Dev BWA")
plt.xlabel("Epoch")
plt.ylabel("BWA")
plt.title(f"BWA curve (hidden_dim={best_dim})")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(working_dir, "bwa_curve_best_hidden_dim.png")
plt.savefig(plot_path)
print(f"Curve saved to {plot_path}")
