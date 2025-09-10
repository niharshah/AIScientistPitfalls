import os, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ----------------------------- 0. I/O & bookkeeping -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {  # dataset name
            "per_lr": {}  # will be filled with sub-dicts keyed by lr
        }
    }
}


# ----------------------------- 1. Helpers ---------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def complexity_weight(sequence: str) -> int:
    return count_color_variety(sequence) + count_shape_variety(sequence)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [complexity_weight(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return float(sum(correct)) / max(1, sum(weights))


# ----------------------------- 2. Dataset loading -------------------------------
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
    shapes, colors = list(string.ascii_uppercase[:6]), list(range(6))

    def make_seq():
        return " ".join(
            random.choice(shapes) + str(random.choice(colors))
            for _ in range(random.randint(4, 9))
        )

    label_rule = lambda seq: sum(int(tok[1]) for tok in seq.split()) % 2

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

# ----------------------------- 3. Vocab + graph encoding ------------------------
vocab = {}
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)


def seq_to_graph(seq, label):
    tokens = seq.split()
    node_idx = [vocab[tok] for tok in tokens]
    edges = [[i, i + 1] for i in range(len(tokens) - 1)] + [
        [i + 1, i] for i in range(len(tokens) - 1)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_idx, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def encode_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, (spr["train"], spr["dev"], spr["test"])
)

batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ----------------------------- 4. Model definition ------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.conv1, self.conv2 = SAGEConv(emb_dim, hidden_dim), SAGEConv(
            hidden_dim, hidden_dim
        )
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(self.emb(x), edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(global_mean_pool(x, batch))


# ----------------------------- 5. Training utilities ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot_samples = 0
    seqs_all, preds_all, labels_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_correct += sum(int(p == y) for p, y in zip(preds, ys))
        tot_samples += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(preds)
        labels_all.extend(ys)
    avg_loss = tot_loss / tot_samples
    acc = tot_correct / tot_samples
    cowa = complexity_weighted_accuracy(seqs_all, labels_all, preds_all)
    return avg_loss, acc, cowa, preds_all, labels_all, seqs_all


# ----------------------------- 6. Hyper-parameter sweep ------------------------
LR_LIST = [5e-4, 1e-3, 2e-3, 5e-3]
EPOCHS = 5
best_lr, best_val_acc = None, -1

for lr in LR_LIST:
    model = GNNClassifier(len(vocab), num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    lr_key = f"{lr:.4g}"
    run_log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
    print(f"\n=========== Training with learning rate {lr} ===========")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_cowa, *_ = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_cowa, *_ = run_epoch(model, dev_loader)
        run_log["losses"]["train"].append(tr_loss)
        run_log["losses"]["val"].append(val_loss)
        run_log["metrics"]["train"].append({"acc": tr_acc, "cowa": tr_cowa})
        run_log["metrics"]["val"].append({"acc": val_acc, "cowa": val_cowa})
        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  val_CoWA={val_cowa:.3f}"
        )
        # track best LR by highest val_acc at final epoch
        if epoch == EPOCHS and val_acc > best_val_acc:
            best_val_acc, best_lr = val_acc, lr
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    test_loss, test_acc, test_cowa, preds, gts, seqs = run_epoch(model, test_loader)
    run_log["predictions"], run_log["ground_truth"], run_log["sequences"] = (
        preds,
        gts,
        seqs,
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["per_lr"][lr_key] = run_log
    print(
        f"TEST (lr={lr_key}) -- loss:{test_loss:.4f} acc:{test_acc:.3f} CoWA:{test_cowa:.3f}"
    )

print(
    f"\nBest learning rate based on validation accuracy: {best_lr}  (val_acc={best_val_acc:.3f})"
)
# ----------------------------- 7. Save results ---------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
