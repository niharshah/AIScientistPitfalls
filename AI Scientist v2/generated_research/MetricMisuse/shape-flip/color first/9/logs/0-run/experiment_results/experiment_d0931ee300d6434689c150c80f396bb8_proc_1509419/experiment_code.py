import os, pathlib, random, time, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import Dataset, DatasetDict, load_dataset

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


############################################################
# ---------- helper: load SPR_BENCH or make toy ----------- #
############################################################
def _load_csv_as_split(csv_path: pathlib.Path):
    return load_dataset(
        "csv",
        data_files=str(csv_path),
        split="train",  # read entire csv as a "train" split
        cache_dir=".cache_dsets",
    )


def load_spr_bench(root: pathlib.Path):
    d = DatasetDict()
    d["train"] = _load_csv_as_split(root / "train.csv")
    d["dev"] = _load_csv_as_split(root / "dev.csv")
    d["test"] = _load_csv_as_split(root / "test.csv")
    return d


def build_toy_dataset(n_train=200, n_dev=50, n_test=50):
    """
    Build a tiny random in-memory dataset for quick debugging.
    Uses datasets.Dataset.from_dict instead of load_dataset to avoid
    passing raw dicts to the file-oriented API (bugfix).
    """
    tokens = ["SC", "TR", "SCr", "SqB", "TrR", "CrR"]

    def rand_seq():
        return " ".join(random.choices(tokens, k=random.randint(4, 9)))

    def rand_lbl():
        return random.randint(0, 2)

    def build_split(n):
        return {
            "id": [str(i) for i in range(n)],
            "sequence": [rand_seq() for _ in range(n)],
            "label": [rand_lbl() for _ in range(n)],
        }

    ds = DatasetDict()
    ds["train"] = Dataset.from_dict(build_split(n_train))
    ds["dev"] = Dataset.from_dict(build_split(n_dev))
    ds["test"] = Dataset.from_dict(build_split(n_test))
    return ds


data_root = pathlib.Path("./SPR_BENCH")
if data_root.exists():
    dsets = load_spr_bench(data_root)
else:
    print("SPR_BENCH not found, generating tiny synthetic dataset.")
    dsets = build_toy_dataset()

num_classes = len(set(dsets["train"]["label"]))
print("Loaded dataset with", num_classes, "classes")


############################################################
# ----------- metrics -------------------------------------#
############################################################
def count_color_variety(seq: str):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def sdwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    corr = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


############################################################
# ---------- graph conversion -----------------------------#
############################################################
def build_vocab(dataset):
    vocab = {"<PAD>": 0}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(dsets["train"])
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def seq_to_graph(sequence: str, label: int):
    toks = sequence.strip().split()
    node_ids = torch.tensor([vocab[t] for t in toks], dtype=torch.long)
    if len(toks) == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        src = list(range(len(toks) - 1)) + list(range(1, len(toks)))
        dst = list(range(1, len(toks))) + list(range(len(toks) - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(
        x=node_ids,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        seq=sequence,  # keep sequence for metrics
    )
    return data


def build_graph_dataset(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs = build_graph_dataset(dsets["train"])
dev_graphs = build_graph_dataset(dsets["dev"])
test_graphs = build_graph_dataset(dsets["test"])


############################################################
# ---------------- model ----------------------------------#
############################################################
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embedding(data.x)  # [N, emb_dim]
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)  # [batch, hidden]
        return self.lin(x)


model = GNNClassifier(vocab_size, 64, 64, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

############################################################
# --------------- experiment container --------------------#
############################################################
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

############################################################
# ----------------- training loop -------------------------#
############################################################
train_loader = GeoLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = GeoLoader(dev_graphs, batch_size=128, shuffle=False)

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    # ---- train ---- #
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.num_graphs
    train_loss = running_loss / len(train_loader.dataset)

    # ---- eval ---- #
    model.eval()
    val_loss, y_true, y_pred, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.to(device))
            val_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1).cpu().tolist()
            labels = batch.y.cpu().tolist()
            seqs.extend(batch.seq)  # sequences retained in batch
            y_true.extend(labels)
            y_pred.extend(preds)
    val_loss /= len(dev_loader.dataset)
    cwa_val = cwa(seqs, y_true, y_pred)
    swa_val = swa(seqs, y_true, y_pred)
    sdwa_val = sdwa(seqs, y_true, y_pred)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SDWA {sdwa_val:.4f} | CWA {cwa_val:.4f} | SWA {swa_val:.4f}"
    )

    # ---- log ---- #
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "sdwa": sdwa_val, "cwa": cwa_val, "swa": swa_val}
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(
        {"epoch": epoch, "loss": train_loss}
    )
    experiment_data["SPR_BENCH"]["losses"]["val"].append(
        {"epoch": epoch, "loss": val_loss}
    )
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

############################################################
# --------------------- test evaluation -------------------#
############################################################
test_loader = GeoLoader(test_graphs, batch_size=128, shuffle=False)
model.eval()
y_true, y_pred, seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds = out.argmax(dim=1).cpu().tolist()
        labels = batch.y.cpu().tolist()
        seqs.extend(batch.seq)
        y_true.extend(labels)
        y_pred.extend(preds)

sdwa_test = sdwa(seqs, y_true, y_pred)
print(f"Test SDWA: {sdwa_test:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
