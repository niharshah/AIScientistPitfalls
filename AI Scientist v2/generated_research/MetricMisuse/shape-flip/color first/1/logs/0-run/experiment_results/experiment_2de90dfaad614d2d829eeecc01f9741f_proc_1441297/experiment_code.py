# ----------------- environment & paths -----------------
import os, pathlib, random, string, time, numpy as np, torch, warnings
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool

warnings.filterwarnings("ignore")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- metric helpers ----------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def harmonic_weighted_accuracy(cwa, swa):
    return 0.0 if (cwa + swa) == 0 else 2 * cwa * swa / (cwa + swa)


# ----------------- data utils --------------------------
def try_load_benchmark():
    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None
    from datasets import load_dataset, DatasetDict

    def _ld(fname):
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def generate_synthetic(nt=200, nd=60, nte=100):
    shapes = list(string.ascii_uppercase[:6])
    colors = list(string.ascii_lowercase[:6])

    def _make(n):
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 15)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seqs.append(" ".join(toks))
            labels.append(int(toks[0][0] == toks[-1][0]))
        return {"sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        train=Dataset.from_dict(_make(nt)),
        dev=Dataset.from_dict(_make(nd)),
        test=Dataset.from_dict(_make(nte)),
    )


dataset = try_load_benchmark()
if dataset is None:
    print("Benchmark not found; generating synthetic data.")
    dataset = generate_synthetic()
num_classes = len(set(dataset["train"]["label"]))
print(f"Dataset sizes: train={len(dataset['train'])}, dev={len(dataset['dev'])}")

# ----------------- vocabulary --------------------------
vocab = {}


def add_tok(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_tok(tok)
print("Vocab size:", len(vocab))


# ----------------- graph conversion --------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    node_ids = torch.tensor([vocab[t] for t in toks], dtype=torch.long)
    if len(toks) > 1:
        src = list(range(len(toks) - 1))
        dst = list(range(1, len(toks)))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    g = Data(
        x=node_ids.unsqueeze(-1),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        seq_raw=seq,
    )
    return g


train_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["train"]["sequence"], dataset["train"]["label"])
]
dev_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["dev"]["sequence"], dataset["dev"]["label"])
]
test_graphs = [
    seq_to_graph(s, l)
    for s, l in zip(dataset["test"]["sequence"], dataset["test"]["label"])
]


# ----------------- model -------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hid=64, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = SAGEConv(embed_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, num_classes)

    def forward(self, data: Batch):
        x = self.embed(data.x.squeeze(-1))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ----------------- experiment tracking -----------------
experiment_data = {"batch_size_tuning": {}}

# ----------------- hyper-parameter search ---------------
batch_sizes = [16, 32, 64, 128]
epochs = 5

for bs in batch_sizes:
    tag = f"bs_{bs}"
    print(f"\n=== Training with batch_size={bs} ===")
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    model = GNNClassifier(len(vocab), num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=max(bs, 64))

    for epoch in range(1, epochs + 1):
        # -------- training ----------
        model.train()
        running_loss, n_graphs = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs
        train_loss = running_loss / n_graphs
        exp["losses"]["train"].append(train_loss)

        # -------- validation --------
        model.eval()
        val_loss, preds, gts, seqs, n_val = 0.0, [], [], [], 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                n_val += batch.num_graphs
                p = logits.argmax(-1).cpu().tolist()
                g = batch.y.cpu().tolist()
                s = batch.seq_raw  # list of raw sequences
                preds.extend(p)
                gts.extend(g)
                seqs.extend(s)
        val_loss /= n_val
        acc = float(np.mean([p == t for p, t in zip(preds, gts)]))
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hwa = harmonic_weighted_accuracy(cwa, swa)

        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["val"].append({"acc": acc, "cwa": cwa, "swa": swa, "hwa": hwa})
        exp["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | acc={acc:.3f} | CWA={cwa:.3f} | "
            f"SWA={swa:.3f} | HWA={hwa:.3f}"
        )

    exp["predictions"] = preds
    exp["ground_truth"] = gts
    experiment_data["batch_size_tuning"][tag] = exp

# ----------------- save result -------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. Saved results to", os.path.join(working_dir, "experiment_data.npy"))
