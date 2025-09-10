import os, pathlib, random, string, time, json, warnings, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------------- working directory & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- metric helpers -------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# -------------------- load or fake data ----------------------
def try_load_benchmark():
    from datasets import load_dataset, DatasetDict

    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


def make_tiny_synthetic(n_tr=300, n_dev=100, n_te=150):
    shapes = list(string.ascii_uppercase[:6])
    colors = list(string.ascii_lowercase[:6])

    def gen(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(4, 12)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seq = " ".join(toks)
            labels.append(int(toks[0][0] == toks[-1][0]))
            seqs.append(seq)
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset

    return {
        k: Dataset.from_dict(gen(v))
        for k, v in [("train", n_tr), ("dev", n_dev), ("test", n_te)]
    }


dataset = try_load_benchmark()
if dataset is None:
    print("SPR_BENCH not found, using synthetic toy data.")
    dataset = make_tiny_synthetic()

num_classes = len(set(dataset["train"]["label"]))
print(
    f"Train={len(dataset['train'])}, Dev={len(dataset['dev'])}, Test={len(dataset['test'])}, Classes={num_classes}"
)

# --------------------- vocabulary ----------------------------
vocab = {}


def add(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add(tok)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ------------------ graph construction -----------------------
def seq_to_graph(seq, label, idx):
    toks = seq.split()
    node_ids = torch.tensor([vocab[t] for t in toks], dtype=torch.long)
    if len(toks) > 1:
        src = list(range(len(toks) - 1))
        dst = list(range(1, len(toks)))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    d = Data(
        x=node_ids.unsqueeze(-1),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        seq_idx=torch.tensor([idx], dtype=torch.long),
    )
    return d


train_seq = dataset["train"]["sequence"]
dev_seq = dataset["dev"]["sequence"]
test_seq = dataset["test"]["sequence"]

train_graphs = [
    seq_to_graph(s, l, i)
    for i, (s, l) in enumerate(zip(train_seq, dataset["train"]["label"]))
]
dev_graphs = [
    seq_to_graph(s, l, i)
    for i, (s, l) in enumerate(zip(dev_seq, dataset["dev"]["label"]))
]
test_graphs = [
    seq_to_graph(s, l, i)
    for i, (s, l) in enumerate(zip(test_seq, dataset["test"]["label"]))
]


# ------------------------- model -----------------------------
class SPRGraphNet(nn.Module):
    def __init__(self, vocab, emb=32, hid=64, n_class=2):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), emb)
        self.conv1 = SAGEConv(emb, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, n_class)

    def forward(self, data):
        x = self.embed(data.x.squeeze()).to(device)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ------------------- experiment tracking ---------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------------- training loop -------------------------
lrs = [3e-4, 1e-3, 3e-3]
epochs = 6
batch_train = 32
batch_eval = 64

for lr in lrs:
    print(f"\n=== LR {lr:.0e} ===")
    model = SPRGraphNet(vocab, n_class=num_classes).to(device)
    optim = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_graphs, batch_size=batch_train, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=batch_eval)
    for ep in range(1, epochs + 1):
        # --- train ---
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optim.step()
            tot_loss += loss.item() * batch.num_graphs
        tr_loss = tot_loss / len(train_loader.dataset)
        experiment_data["SPR"]["losses"]["train"].append(tr_loss)
        # --- eval ---
        model.eval()
        dev_loss = 0.0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = loss_fn(logits, batch.y)
                dev_loss += loss.item() * batch.num_graphs
                pr = logits.argmax(-1).cpu().tolist()
                gt = batch.y.cpu().tolist()
                idxs = batch.seq_idx.cpu().tolist()
                preds.extend(pr)
                gts.extend(gt)
                seqs.extend([dev_seq[i] for i in idxs])
        dev_loss /= len(dev_loader.dataset)
        acc = float(np.mean([p == t for p, t in zip(preds, gts)]))
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hpa = harmonic_poly_accuracy(cwa, swa)
        experiment_data["SPR"]["losses"]["val"].append(dev_loss)
        experiment_data["SPR"]["metrics"]["val"].append(
            {"acc": acc, "cwa": cwa, "swa": swa, "hpa": hpa}
        )
        print(
            f"Epoch {ep}: validation_loss = {dev_loss:.4f}  Acc={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} HPA={hpa:.3f}"
        )

# --------------- save everything -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
