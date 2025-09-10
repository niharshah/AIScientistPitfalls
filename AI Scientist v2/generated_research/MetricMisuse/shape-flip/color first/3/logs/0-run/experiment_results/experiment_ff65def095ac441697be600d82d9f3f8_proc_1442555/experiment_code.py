import os, pathlib, time, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import List
from datasets import DatasetDict, load_dataset

# ---------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# mandatory device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# ---------- Helper to locate the dataset -----------------------------
def locate_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in candidates:
        if p and (p / "train.csv").exists() and (p / "dev.csv").exists():
            print(f"Found SPR_BENCH at: {p.resolve()}")
            return p.resolve()
    raise FileNotFoundError(
        "SPR_BENCH dataset not found. Place the folder in the current or parent "
        "directory or set the environment variable SPR_DATA_PATH to its location."
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) > 0 else 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) > 0 else 1)


# ---------------------------------------------------------------------
# ---------- Load dataset ---------------------------------------------
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)


# ---------------------------------------------------------------------
# ---------- Build vocabularies ---------------------------------------
def extract_tokens(seq: str) -> List[str]:
    return seq.strip().split()


token_set, label_set = set(), set()
for ex in spr["train"]:
    token_set.update(extract_tokens(ex["sequence"]))
    label_set.add(ex["label"])

token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
label2idx = {lab: i for i, lab in enumerate(sorted(label_set))}
idx2label = {i: lab for lab, i in label2idx.items()}


# ---------------------------------------------------------------------
# ---------- Graph construction ---------------------------------------
def seq_to_data(example):
    seq = example["sequence"]
    tokens = extract_tokens(seq)
    node_indices = [token2idx.get(tok, 0) for tok in tokens]
    x = torch.tensor(node_indices, dtype=torch.long).unsqueeze(-1)

    if len(tokens) > 1:
        src = torch.arange(0, len(tokens) - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
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
train_loader_full = DataLoader(train_graphs, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------
# ---------- Model -----------------------------------------------------
class SPR_GCN(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, num_classes: int, dropout_rate: float
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.gcn1 = GCNConv(embed_dim, 64)
        self.gcn2 = GCNConv(64, 64)
        self.lin = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.embed(x.squeeze(-1))
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.lin(x)


# ---------------------------------------------------------------------
# ---------- Evaluation function --------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=-1).cpu().tolist()
            labels = batch.y.view(-1).cpu().tolist()
            seqs = batch.seq
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(seqs)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    bwa = (cwa + swa) / 2.0
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, bwa, cwa, swa, all_preds, all_labels


# ---------------------------------------------------------------------
# ---------- Hyperparameter tuning over dropout -----------------------
experiment_data = {"dropout_rate": {"SPR_BENCH": {}}}
dropout_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
num_epochs = 5
best_dev_bwa, best_rate, best_state = -1.0, None, None

for dr in dropout_grid:
    print(f"\n=== Training with dropout={dr:.2f} ===")
    model = SPR_GCN(len(token2idx), 32, len(label2idx), dropout_rate=dr).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    rate_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

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

        val_loss, val_bwa, val_cwa, val_swa, _, _ = evaluate(
            model, dev_loader, criterion
        )
        _, train_bwa, _, _, _, _ = evaluate(model, train_loader_full, criterion)

        rate_dict["losses"]["train"].append(train_loss)
        rate_dict["losses"]["val"].append(val_loss)
        rate_dict["metrics"]["train"].append(train_bwa)
        rate_dict["metrics"]["val"].append(val_bwa)
        rate_dict["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"BWA={val_bwa:.4f} (CWA={val_cwa:.4f}, SWA={val_swa:.4f})"
        )

    # keep best model w.r.t dev BWA
    if val_bwa > best_dev_bwa:
        best_dev_bwa = val_bwa
        best_rate = dr
        best_state = model.state_dict()

    experiment_data["dropout_rate"]["SPR_BENCH"][f"dr_{dr:.2f}"] = rate_dict
    # free cuda mem before next run
    del model
    torch.cuda.empty_cache()

print(f"\nBest dev BWA {best_dev_bwa:.4f} achieved with dropout={best_rate:.2f}")

# ---------------------------------------------------------------------
# ---------- Final test evaluation with best model --------------------
best_model = SPR_GCN(len(token2idx), 32, len(label2idx), dropout_rate=best_rate).to(
    device
)
best_model.load_state_dict(best_state)
criterion = nn.CrossEntropyLoss()
test_loss, test_bwa, test_cwa, test_swa, test_preds, test_labels = evaluate(
    best_model, test_loader, criterion
)
print(
    f"Test -> loss: {test_loss:.4f}  BWA: {test_bwa:.4f}  CWA: {test_cwa:.4f}  SWA: {test_swa:.4f}"
)

# store final test results
experiment_data["dropout_rate"]["SPR_BENCH"]["best_rate"] = best_rate
experiment_data["dropout_rate"]["SPR_BENCH"]["test"] = {
    "loss": test_loss,
    "BWA": test_bwa,
    "CWA": test_cwa,
    "SWA": test_swa,
    "predictions": test_preds,
    "ground_truth": test_labels,
}

# ---------------------------------------------------------------------
# ---------- Plotting best run ----------------------------------------
epochs = np.arange(1, num_epochs + 1)
best_dict = experiment_data["dropout_rate"]["SPR_BENCH"][f"dr_{best_rate:.2f}"]
plt.figure()
plt.plot(epochs, best_dict["metrics"]["train"], label="Train BWA")
plt.plot(epochs, best_dict["metrics"]["val"], label="Dev BWA")
plt.xlabel("Epoch")
plt.ylabel("BWA")
plt.title(f"BWA over epochs (dropout={best_rate:.2f})")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(working_dir, f"bwa_curve_spr_dropout_{best_rate:.2f}.png")
plt.savefig(plot_path)
print(f"Curve saved to {plot_path}")

# ---------------------------------------------------------------------
# ---------- Save metrics ---------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
