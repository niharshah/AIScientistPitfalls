import os, random, string, time

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


############################################################
# 1. Helpers for SPR logic and evaluation
############################################################
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


############################################################
# 2. Load real dataset if available, else make synthetic data
############################################################
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")  # user can set path


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
    # ---------- Synthetic fallback ----------
    print("No SPR_BENCH found; generating synthetic toy dataset")
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colors = list(range(6))  # 0-5

    def make_seq():
        L = random.randint(4, 9)
        toks = [random.choice(shapes) + str(random.choice(colors)) for _ in range(L)]
        return " ".join(toks)

    def label_rule(seq):
        # parity of total color ids as a trivial hidden "rule"
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

############################################################
# 3. Token vocabulary & graph conversion
############################################################
# Build vocabulary of token strings from training data
vocab = {}


def add_token(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        add_token(tok)
vocab_size = len(vocab)
pad_index = vocab_size  # not used actually


def seq_to_graph(seq, label):
    tokens = seq.split()
    node_idx = [vocab[tok] for tok in tokens]
    edge_index = []
    for i in range(len(tokens) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_idx, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def encode_split(split_ds):
    return [
        seq_to_graph(seq, lab)
        for seq, lab in zip(split_ds["sequence"], split_ds["label"])
    ]


train_graphs = encode_split(spr["train"])
dev_graphs = encode_split(spr["dev"])
test_graphs = encode_split(spr["test"])

############################################################
# 4. DataLoaders
############################################################
batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


############################################################
# 5. Model
############################################################
class GNNClassifier(nn.Module):
    def __init__(self, vocab, emb_dim=64, hidden_dim=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.emb(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = GNNClassifier(vocab, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)


############################################################
# 6. Training / Evaluation utilities
############################################################
def run_epoch(loader, train=False):
    total_loss, total_correct, total_samples = 0, 0, 0
    seqs_all, preds_all, labels_all = [], [], []
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        total_correct += sum(int(p == y) for p, y in zip(preds, ys))
        total_samples += batch.num_graphs
        seqs_all.extend(batch.seq)
        preds_all.extend(preds)
        labels_all.extend(ys)
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    cowa = complexity_weighted_accuracy(seqs_all, labels_all, preds_all)
    return avg_loss, acc, cowa, preds_all, labels_all, seqs_all


############################################################
# 7. Experiment bookkeeping
############################################################
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
}

############################################################
# 8. Train loop
############################################################
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_cowa, *_ = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_cowa, *_ = run_epoch(dev_loader, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"acc": tr_acc, "cowa": tr_cowa}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "cowa": val_cowa}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_acc={val_acc:.3f}, val_CoWA={val_cowa:.3f}"
    )

############################################################
# 9. Test evaluation
############################################################
test_loss, test_acc, test_cowa, preds, gts, seqs = run_epoch(test_loader, train=False)
print(f"TEST -- loss: {test_loss:.4f}, acc: {test_acc:.3f}, CoWA: {test_cowa:.3f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["sequences"] = seqs
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
