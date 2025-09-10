import os, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------- I/O setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- Helper metrics ----------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weight(seq: str) -> int:
    return count_color_variety(seq) + count_shape_variety(seq)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return float(sum(corr)) / max(1, sum(w))


# ---------- Load / build SPR dataset ----------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path: str) -> DatasetDict:
    if os.path.isdir(path):

        def _csv(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        print("Loading real SPR_BENCH from", path)
        return DatasetDict(train=_csv("train"), dev=_csv("dev"), test=_csv("test"))
    # synthetic fallback
    print("No SPR_BENCH found; generating synthetic toy dataset")
    shapes = list(string.ascii_uppercase[:6])
    colors = list(range(6))

    def make_seq():
        L = random.randint(4, 9)
        toks = [random.choice(shapes) + str(random.choice(colors)) for _ in range(L)]
        return " ".join(toks)

    def label_rule(seq):
        return sum(int(tok[1]) for tok in seq.split()) % 2

    def build_split(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [label_rule(s) for s in seqs],
            }
        )

    return DatasetDict(
        train=build_split(800), dev=build_split(200), test=build_split(200)
    )


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# ---------- Vocabulary ----------
vocab = {}


def add_token(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
vocab_size = len(vocab)


# ---------- Sequence -> graph ----------
def seq_to_graph(seq, label):
    tokens = seq.split()
    node_idx = [vocab[tok] for tok in tokens]
    edges = []
    for i in range(len(tokens) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_idx, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def encode_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, [spr["train"], spr["dev"], spr["test"]]
)

# ---------- DataLoaders ----------
batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ---------- Model ----------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, num_layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList()
        in_dim = emb_dim
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.emb(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------- Training utilities ----------
def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = tot_corr = tot_samp = 0
    seqs_all = []
    preds_all = []
    labels_all = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_corr += sum(int(p == y) for p, y in zip(pred, ys))
        tot_samp += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(pred)
        labels_all.extend(ys)
    avg = tot_loss / tot_samp
    acc = tot_corr / tot_samp
    cowa = complexity_weighted_accuracy(seqs_all, labels_all, preds_all)
    return avg, acc, cowa, preds_all, labels_all, seqs_all


# ---------- Experiment bookkeeping ----------
experiment_data = {"num_gnn_layers": {}}

# ---------- Hyperparameter sweep ----------
depths = [1, 2, 3, 4]
EPOCHS = 5
for depth in depths:
    print(f"\n=== Training model with {depth} GNN layer(s) ===")
    model = GNNClassifier(
        len(vocab), emb_dim=64, hidden_dim=64, num_classes=num_classes, num_layers=depth
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_cowa, *_ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, val_cowa, *_ = run_epoch(model, dev_loader, criterion)
        record["losses"]["train"].append(tr_loss)
        record["losses"]["val"].append(val_loss)
        record["metrics"]["train"].append({"acc": tr_acc, "cowa": tr_cowa})
        record["metrics"]["val"].append({"acc": val_acc, "cowa": val_cowa})
        print(
            f"Depth {depth} Epoch {epoch}: val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.3f}, val_CoWA={val_cowa:.3f}"
        )
    # Test evaluation
    tst_loss, tst_acc, tst_cowa, preds, gts, seqs = run_epoch(
        model, test_loader, criterion
    )
    record["test"] = {"loss": tst_loss, "acc": tst_acc, "cowa": tst_cowa}
    record["predictions"] = preds
    record["ground_truth"] = gts
    record["sequences"] = seqs
    print(
        f"Depth {depth} TEST -- loss:{tst_loss:.4f}, acc:{tst_acc:.3f}, CoWA:{tst_cowa:.3f}"
    )
    experiment_data["num_gnn_layers"][f"SPR_BENCH_layers_{depth}"] = record

# ---------- Save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
