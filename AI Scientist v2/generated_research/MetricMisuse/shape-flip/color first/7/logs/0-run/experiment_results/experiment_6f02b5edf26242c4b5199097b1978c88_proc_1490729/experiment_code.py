import os, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- Helpers for SPR --------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split()))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split()))


def complexity_weight(seq: str) -> int:
    return count_color_variety(seq) + count_shape_variety(seq)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return float(sum(good)) / max(1, sum(w))


# -------------------- Load / synthesize SPR dataset --------------------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path: str) -> DatasetDict:
    if os.path.isdir(path):
        print("Loading real SPR_BENCH from", path)

        def _csv(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict(train=_csv("train"), dev=_csv("dev"), test=_csv("test"))
    # synthetic fallback
    print("No SPR_BENCH found; generating synthetic data")
    shapes, colors = list(string.ascii_uppercase[:6]), list(range(6))

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + str(random.choice(colors)) for _ in range(L)
        )

    def label_rule(seq):  # parity of color ids
        return sum(int(tok[1]) for tok in seq.split()) % 2

    def build(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [label_rule(s) for s in seqs],
            }
        )

    return DatasetDict(train=build(800), dev=build(200), test=build(200))


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# -------------------- Vocabulary & graphs --------------------
vocab = {}


def add_tok(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add_tok(tok)
vocab_size, pad_index = len(vocab), len(vocab)


def seq_to_graph(seq, label):
    toks = seq.split()
    node_idx = [vocab[t] for t in toks]
    e = [[i, i + 1] for i in range(len(toks) - 1)] + [
        [i + 1, i] for i in range(len(toks) - 1)
    ]
    edge_index = torch.tensor(e, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_idx, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def encode(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode, (spr["train"], spr["dev"], spr["test"])
)

batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# -------------------- Model --------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab, emb_dim=64, hidden_dim=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge, batch = data.x, data.edge_index, data.batch
        x = self.emb(x)
        x = F.relu(self.conv1(x, edge))
        x = F.relu(self.conv2(x, edge))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# -------------------- Train / eval --------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, train_mode=False, optim=None):
    model.train() if train_mode else model.eval()
    tot_loss = tot_correct = tot = 0
    seqs_all, preds_all, labels_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * batch.num_graphs
        pred = out.argmax(1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_correct += sum(int(p == y) for p, y in zip(pred, ys))
        tot += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(pred)
        labels_all.extend(ys)
    avg_loss = tot_loss / tot
    acc = tot_correct / tot
    cowa = complexity_weighted_accuracy(seqs_all, labels_all, preds_all)
    return avg_loss, acc, cowa, preds_all, labels_all, seqs_all


# -------------------- Hyperparameter sweep --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"hidden_dim": {"SPR_BENCH": {}}}

hidden_dims = [32, 64, 128, 256]
EPOCHS = 5

for hd in hidden_dims:
    print(f"\n=== Training with hidden_dim={hd} ===")
    model = GNNClassifier(vocab, hidden_dim=hd, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_cowa, *_ = run_epoch(model, train_loader, True, optimizer)
        vl_loss, vl_acc, vl_cowa, *_ = run_epoch(model, dev_loader)
        exp_rec["losses"]["train"].append(tr_loss)
        exp_rec["losses"]["val"].append(vl_loss)
        exp_rec["metrics"]["train"].append({"acc": tr_acc, "cowa": tr_cowa})
        exp_rec["metrics"]["val"].append({"acc": vl_acc, "cowa": vl_cowa})
        print(
            f"  Epoch {epoch}: val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} val_cowa={vl_cowa:.3f}"
        )
    # final test
    ts_loss, ts_acc, ts_cowa, preds, gts, seqs = run_epoch(model, test_loader)
    exp_rec["test"] = {"loss": ts_loss, "acc": ts_acc, "cowa": ts_cowa}
    exp_rec["predictions"] = preds
    exp_rec["ground_truth"] = gts
    exp_rec["sequences"] = seqs
    experiment_data["hidden_dim"]["SPR_BENCH"][str(hd)] = exp_rec
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  TEST  -- loss={ts_loss:.4f} acc={ts_acc:.3f} cowa={ts_cowa:.3f}")

# -------------------- Save everything --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
