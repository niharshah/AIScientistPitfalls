import os, pathlib, random, string, time, numpy as np, torch, warnings
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from datasets import load_dataset, Dataset, DatasetDict

# ------------------- mandatory boiler-plate -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

warnings.filterwarnings("ignore")


# ------------------- metrics ---------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if len(t) > 0))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / max(cwa + swa, 1e-9)


# ------------------- data utilities --------------------------
def try_load_spr_bench():
    root = pathlib.Path("./SPR_BENCH")
    if not root.exists():
        return None

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
            length = random.randint(4, 15)
            toks = [
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            ]
            seqs.append(" ".join(toks))
            labels.append(int(toks[0][0] == toks[-1][0]))
        return {"sequence": seqs, "label": labels}

    return DatasetDict(
        train=Dataset.from_dict(_make(nt)),
        dev=Dataset.from_dict(_make(nd)),
        test=Dataset.from_dict(_make(nte)),
    )


dataset = try_load_spr_bench()
if dataset is None:
    print("Benchmark not found; using synthetic dataset.")
    dataset = generate_synthetic()

num_classes = len(set(dataset["train"]["label"]))
print(
    f"Dataset sizes: train={len(dataset['train'])}, dev={len(dataset['dev'])}, classes={num_classes}"
)

# ------------------- vocabulary ------------------------------
vocab = {}


def add_token(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in dataset["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
print("Vocabulary size:", len(vocab))


# ------------------- graph conversion ------------------------
def seq_to_graph(seq, label):
    toks = seq.split()
    node_ids = torch.tensor([vocab[t] for t in toks], dtype=torch.long)
    if len(toks) > 1:
        src = list(range(len(toks) - 1))
        dst = list(range(1, len(toks)))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    g = Data(x=node_ids.unsqueeze(-1), edge_index=edge_index, y=torch.tensor([label]))
    g.seq_raw = seq
    return g


def build_graph_list(split):
    return [
        seq_to_graph(s, l)
        for s, l in zip(dataset[split]["sequence"], dataset[split]["label"])
    ]


train_graphs = build_graph_list("train")
dev_graphs = build_graph_list("dev")


# ------------------- model -----------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hid=64, classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = SAGEConv(embed_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ------------------- experiment tracking ---------------------
experiment_data = {
    "batch_size_tuning": {  # single dataset bucket
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------- training / tuning -----------------------
batch_sizes = [16, 32, 64, 128]
epochs = 5

for bs in batch_sizes:
    print(f"\n===== Training with batch_size={bs} =====")
    model = GNNClassifier(len(vocab), classes=num_classes).to(device)  # bug fixed
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=max(64, bs))
    for ep in range(1, epochs + 1):
        # ---- training ----
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
        tr_loss = tot_loss / len(train_loader.dataset)
        experiment_data["batch_size_tuning"]["losses"]["train"].append(tr_loss)

        # ---- validation ----
        model.eval()
        val_loss, preds, gts, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                p = logits.argmax(-1).cpu().tolist()
                g = batch.y.cpu().tolist()
                preds.extend(p)
                gts.extend(g)
                seqs.extend(batch.seq_raw)
        val_loss /= len(dev_loader.dataset)
        acc = float(np.mean([p == t for p, t in zip(preds, gts)]))
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hwa = harmonic_weighted_accuracy(cwa, swa)

        experiment_data["batch_size_tuning"]["losses"]["val"].append(val_loss)
        experiment_data["batch_size_tuning"]["metrics"]["val"].append(
            {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "hwa": hwa}
        )
        experiment_data["batch_size_tuning"]["timestamps"].append(time.time())
        print(
            f"Epoch {ep}/{epochs}: validation_loss = {val_loss:.4f} | acc={acc:.3f} | CWA={cwa:.3f} | SWA={swa:.3f} | HWA={hwa:.3f}"
        )

    experiment_data["batch_size_tuning"]["predictions"].append(preds)
    experiment_data["batch_size_tuning"]["ground_truth"].append(gts)

# ------------------- save results ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    f"\nFinished. Results saved to {os.path.join(working_dir, 'experiment_data.npy')}"
)
