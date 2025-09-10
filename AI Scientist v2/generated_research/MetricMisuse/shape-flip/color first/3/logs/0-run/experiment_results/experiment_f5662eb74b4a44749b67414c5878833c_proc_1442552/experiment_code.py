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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
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
    raise FileNotFoundError("SPR_BENCH dataset not found.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):  # helper
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


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


# ------------------------------------------------------------
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)


def extract_tokens(seq: str) -> List[str]:
    return seq.strip().split()


token_set, label_set = set(), set()
for ex in spr["train"]:
    token_set.update(extract_tokens(ex["sequence"]))
    label_set.add(ex["label"])
token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
label2idx = {lab: i for i, lab in enumerate(sorted(label_set))}
idx2label = {i: lab for lab, i in label2idx.items()}


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
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ------------------------------------------------------------
class SPR_GCN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.gcn1 = GCNConv(embed_dim, 64)
        self.gcn2 = GCNConv(64, 64)
        self.lin = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.embed(x.squeeze(-1))
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.drop(x)
        return self.lin(x)


model = SPR_GCN(len(token2idx), 32, len(label2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------
experiment_data = {
    "num_epochs": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}


# ------------------------------------------------------------
def evaluate(loader):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(-1).cpu().tolist()
            labels = batch.y.view(-1).cpu().tolist()
            seqs = batch.seq
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(seqs)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    bwa = (cwa + swa) / 2.0
    return total_loss / len(loader.dataset), bwa, cwa, swa, all_preds, all_labels


# ------------------------------------------------------------
max_epochs = 30  # increased budget
patience = 5  # early stopping patience
best_val_bwa = -1.0
epochs_no_improve = 0
actual_epochs_run = 0

for epoch in range(1, max_epochs + 1):
    actual_epochs_run += 1
    model.train()
    accum_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        accum_loss += loss.item() * batch.num_graphs
    train_loss = accum_loss / len(train_loader.dataset)

    val_loss, val_bwa, val_cwa, val_swa, _, _ = evaluate(dev_loader)
    _, train_bwa, _, _, _, _ = evaluate(train_loader)

    # logging
    edict = experiment_data["num_epochs"]["SPR_BENCH"]
    edict["losses"]["train"].append(train_loss)
    edict["losses"]["val"].append(val_loss)
    edict["metrics"]["train"].append(train_bwa)
    edict["metrics"]["val"].append(val_bwa)
    edict["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"BWA={val_bwa:.4f} (CWA={val_cwa:.4f}, SWA={val_swa:.4f})"
    )

    # early stopping logic
    if val_bwa > best_val_bwa + 1e-6:
        best_val_bwa = val_bwa
        epochs_no_improve = 0
        best_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(
                f"No improvement for {patience} epochs; early stopping at epoch {epoch}."
            )
            break

# ------------------------------------------------------------
print(
    f"Training finished after {actual_epochs_run} epochs; best dev BWA = {best_val_bwa:.4f}"
)
# Load best weights before final test evaluation
model.load_state_dict(best_state)

epochs = np.arange(
    1, len(experiment_data["num_epochs"]["SPR_BENCH"]["metrics"]["val"]) + 1
)
plt.figure()
plt.plot(
    epochs,
    experiment_data["num_epochs"]["SPR_BENCH"]["metrics"]["train"],
    label="Train BWA",
)
plt.plot(
    epochs,
    experiment_data["num_epochs"]["SPR_BENCH"]["metrics"]["val"],
    label="Dev BWA",
)
plt.xlabel("Epoch")
plt.ylabel("BWA")
plt.title("BWA over epochs (tuned num_epochs)")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(working_dir, "bwa_curve_spr.png")
plt.savefig(plot_path)
print(f"Curve saved to {plot_path}")

# ------------------------------------------------------------
test_loss, test_bwa, test_cwa, test_swa, test_preds, test_labels = evaluate(test_loader)
print(
    f"Final Test -> loss: {test_loss:.4f}  BWA: {test_bwa:.4f}  "
    f"CWA: {test_cwa:.4f}  SWA: {test_swa:.4f}"
)

# store predictions
edict = experiment_data["num_epochs"]["SPR_BENCH"]
edict["predictions"] = test_preds
edict["ground_truth"] = test_labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
