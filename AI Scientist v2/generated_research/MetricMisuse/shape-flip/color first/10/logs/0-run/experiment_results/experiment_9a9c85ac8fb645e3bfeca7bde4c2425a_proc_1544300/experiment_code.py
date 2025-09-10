import os, pathlib, random, time, math, sys, itertools
import numpy as np
import torch, warnings
from torch import nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from datasets import Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ----------------- experiment store -----------------
experiment_data = {
    "num_gnn_layers": {"SPR_BENCH": {}}  # each layer setting will be stored here
}

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- metrics helpers -----------------
def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split()))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ----------------- dataset processing -----------------
UNK = "<UNK>"


def build_vocab(sequences):
    vocab = {tok for seq in sequences for tok in seq.split()}
    token2idx = {tok: i + 1 for i, tok in enumerate(sorted(vocab))}
    token2idx[UNK] = 0
    return token2idx


def sequence_to_graph(seq, token2idx, label_idx):
    tokens = seq.strip().split()
    n = len(tokens)
    ids = [token2idx.get(t, token2idx[UNK]) for t in tokens]
    if n == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        src = torch.arange(0, n - 1, dtype=torch.long)
        dst = torch.arange(1, n, dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    x = torch.tensor(ids, dtype=torch.long)
    y = torch.tensor([label_idx], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def prepare_graph_datasets(spr):
    token2idx = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(labels)}

    def _convert(split):
        return [
            sequence_to_graph(seq, token2idx, label2idx[lab])
            for seq, lab in zip(spr[split]["sequence"], spr[split]["label"])
        ]

    return {s: _convert(s) for s in ["train", "dev", "test"]}, token2idx, label2idx


# ----------------- model -----------------
class GraphClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        convs = []
        if num_layers >= 1:
            convs.append(SAGEConv(emb_dim, hidden_dim))
            for _ in range(num_layers - 1):
                convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList(convs)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ----------------- load dataset -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
print(f"Looking for data at: {DATA_PATH}")
if DATA_PATH.exists():
    from datasets import load_dataset

    def _ld(f):  # helper
        return load_dataset(
            "csv",
            data_files=str(DATA_PATH / f),
            split="train",
            cache_dir=".cache_dsets",
        )

    spr_bench = DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )
else:
    print("Real dataset not found. Using synthetic placeholder.")

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labs = [], []
        for i in range(n):
            L = random.randint(4, 8)
            tokens = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seqs.append(" ".join(tokens))
            labs.append(random.choice(["yes", "no"]))
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    spr_bench = DatasetDict(
        {
            "train": Dataset.from_dict(synth(300)),
            "dev": Dataset.from_dict(synth(80)),
            "test": Dataset.from_dict(synth(80)),
        }
    )

graph_sets, token2idx, label2idx = prepare_graph_datasets(spr_bench)
num_classes = len(label2idx)
print(f"Vocab size: {len(token2idx)} | #classes: {num_classes}")

train_loader = DataLoader(graph_sets["train"], batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_sets["dev"], batch_size=128)
test_loader = DataLoader(graph_sets["test"], batch_size=128)

# ----------------- hyperparameter search -----------------
LAYER_CHOICES = [1, 2, 3, 4]
EPOCHS = 5
for n_layers in LAYER_CHOICES:
    print(f"\n===== Training model with {n_layers} GraphSAGE layer(s) =====")
    model = GraphClassifier(
        vocab_size=len(token2idx),
        emb_dim=32,
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=n_layers,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # prepare storage dict
    exp_rec = {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- training ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(graph_sets["train"])
        exp_rec["losses"]["train"].append(train_loss)

        # ---- validation ----
        model.eval()
        val_loss, all_preds, all_labels, all_seqs = 0.0, [], [], []
        with torch.no_grad():
            for b_idx, batch in enumerate(dev_loader):
                seqs = spr_bench["dev"]["sequence"][
                    b_idx * 128 : b_idx * 128 + batch.num_graphs
                ]
                batch = batch.to(device)
                logits = model(batch)
                loss = cross_entropy(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=-1).cpu().tolist()
                labs = batch.y.cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs)
                all_seqs.extend(seqs)
        val_loss /= len(graph_sets["dev"])
        exp_rec["losses"]["val"].append(val_loss)

        inv_label = {v: k for k, v in label2idx.items()}
        pred_lbls = [inv_label[p] for p in all_preds]
        true_lbls = [inv_label[t] for t in all_labels]
        cwa = color_weighted_accuracy(all_seqs, true_lbls, pred_lbls)
        swa = shape_weighted_accuracy(all_seqs, true_lbls, pred_lbls)
        hwa = harmonic_weighted_accuracy(cwa, swa)
        exp_rec["metrics"]["val"].append({"cwa": cwa, "swa": swa, "hwa": hwa})
        exp_rec["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}"
        )

    # ---- final test ----
    test_preds, test_labels, test_seqs = [], [], []
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            seqs = spr_bench["test"]["sequence"][
                b_idx * 128 : b_idx * 128 + batch.num_graphs
            ]
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1).cpu().tolist()
            labs = batch.y.cpu().tolist()
            test_preds.extend(preds)
            test_labels.extend(labs)
            test_seqs.extend(seqs)
    inv_label = {v: k for k, v in label2idx.items()}
    pred_lbls = [inv_label[p] for p in test_preds]
    true_lbls = [inv_label[t] for t in test_labels]
    cwa_t = color_weighted_accuracy(test_seqs, true_lbls, pred_lbls)
    swa_t = shape_weighted_accuracy(test_seqs, true_lbls, pred_lbls)
    hwa_t = harmonic_weighted_accuracy(cwa_t, swa_t)
    exp_rec["metrics"]["test"] = {"cwa": cwa_t, "swa": swa_t, "hwa": hwa_t}
    exp_rec["predictions"] = pred_lbls
    exp_rec["ground_truth"] = true_lbls
    print(f"Test CWA={cwa_t:.3f} SWA={swa_t:.3f} HWA={hwa_t:.3f}")

    # store results
    experiment_data["num_gnn_layers"]["SPR_BENCH"][n_layers] = exp_rec

# ----------------- save -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Experiment data saved to {os.path.join(working_dir,'experiment_data.npy')}")
