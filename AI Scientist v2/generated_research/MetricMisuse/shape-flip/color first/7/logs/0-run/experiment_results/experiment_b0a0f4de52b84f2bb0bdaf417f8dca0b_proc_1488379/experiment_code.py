import os, sys, pathlib, random, time, math, numpy as np, torch
from torch import nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# required working directory & device handling
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ------------------------------------------------------------------
# try to import helper that loads real SPR_BENCH; if not possible build synthetic set
try:
    from SPR import load_spr_bench, count_color_variety, count_shape_variety
except Exception as e:
    print("Could not import SPR helper, creating synthetic utilities.", e)

    def count_color_variety(sequence: str) -> int:
        return len(set(tok[1] for tok in sequence.split()))

    def count_shape_variety(sequence: str) -> int:
        return len(set(tok[0] for tok in sequence.split()))

    def load_spr_bench(_root):
        # create very small synthetic dsets with 3 labels
        import datasets

        def synth_split(n):
            seqs, labels = [], []
            shapes = list("ABCDE")
            colors = list("12345")
            for i in range(n):
                length = random.randint(4, 12)
                seq = " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(length)
                )
                label = random.randint(0, 2)
                seqs.append(seq)
                labels.append(label)
            return datasets.Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        d = {
            "train": synth_split(400),
            "dev": synth_split(120),
            "test": synth_split(120),
        }
        return datasets.DatasetDict(d)


# ------------------------------------------------------------------
# load data
DATA_PATH = pathlib.Path("./SPR_BENCH")
spr_bench = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr_bench.keys())


# ------------------------------------------------------------------
# Build vocabulary of tokens
def build_vocab(dataset):
    vocab = {}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1  # reserve 0 for UNK
    return vocab


vocab = build_vocab(spr_bench["train"])
unk_idx = 0
vocab_size = len(vocab) + 1
print(f"Vocab size: {vocab_size}")

# ------------------------------------------------------------------
# Graph construction utilities
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def seq_to_graph(sequence, label):
    toks = sequence.strip().split()
    n = len(toks)
    x = torch.tensor([vocab.get(t, unk_idx) for t in toks], dtype=torch.long)
    # edges between consecutive tokens (bidirectional)
    src = torch.arange(0, n - 1, dtype=torch.long)
    dst = torch.arange(1, n, dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        sequence=sequence,
    )


def make_graph_list(split_ds):
    return [
        seq_to_graph(seq, lbl)
        for seq, lbl in zip(split_ds["sequence"], split_ds["label"])
    ]


train_graphs = make_graph_list(spr_bench["train"])
dev_graphs = make_graph_list(spr_bench["dev"])
test_graphs = make_graph_list(spr_bench["test"])

# ------------------------------------------------------------------
# PyG GCN model
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.emb(data.x).to(device)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)  # [num_graphs, hidden_dim]
        return self.lin(x)


num_classes = len(set(spr_bench["train"]["label"]))
model = GCNClassifier(
    vocab_size, emb_dim=32, hidden_dim=64, num_classes=num_classes
).to(device)

# ------------------------------------------------------------------
# optimizer AFTER model moved to device
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ------------------------------------------------------------------
# experiment_data structure
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_CoWA": [], "val_CoWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epoch_time": [],
    }
}


# ------------------------------------------------------------------
def complexity_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) + count_shape_variety(seq) for seq in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    tot_w = sum(weights)
    return sum(correct) / tot_w if tot_w > 0 else 0.0


# data loaders
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)


# ------------------------------------------------------------------
def run_epoch(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    epoch_loss, preds, trues, seqs = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = F.cross_entropy(out, batch.y.view(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1).detach().cpu().tolist()
        true = batch.y.view(-1).cpu().tolist()
        preds.extend(pred)
        trues.extend(true)
        seqs.extend(batch.sequence)
    epoch_loss /= len(loader.dataset)
    cowa = complexity_weighted_accuracy(seqs, trues, preds)
    return epoch_loss, cowa, preds, trues


# ------------------------------------------------------------------
num_epochs = 5
best_val_cowa = -1
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss, train_cowa, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_cowa, val_preds, val_trues = run_epoch(dev_loader, train=False)
    dt = time.time() - t0
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_CoWA = {val_cowa:.4f}"
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_CoWA"].append(train_cowa)
    experiment_data["SPR_BENCH"]["metrics"]["val_CoWA"].append(val_cowa)
    experiment_data["SPR_BENCH"]["epoch_time"].append(dt)

    # store best epoch predictions
    if val_cowa > best_val_cowa:
        best_val_cowa = val_cowa
        experiment_data["SPR_BENCH"]["predictions"] = val_preds
        experiment_data["SPR_BENCH"]["ground_truth"] = val_trues

# ------------------------------------------------------------------
# final test evaluation
test_loss, test_cowa, test_preds, test_trues = run_epoch(test_loader, train=False)
print(f"Final Test - loss: {test_loss:.4f}, CoWA: {test_cowa:.4f}")
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_CoWA"] = test_cowa

# ------------------------------------------------------------------
# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
