import os, pathlib, random, time, math, sys, itertools
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from datasets import DatasetDict
from typing import List, Dict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ----------------- experiment log -----------------
experiment_data = {}

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- helper: load dataset -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


# ----------------- metrics -----------------
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


def build_vocab(sequences: List[str]) -> Dict[str, int]:
    vocab = {token for seq in sequences for token in seq.split()}
    token2idx = {tok: i + 1 for i, tok in enumerate(sorted(vocab))}
    token2idx[UNK] = 0
    return token2idx


def sequence_to_graph(seq: str, token2idx: Dict[str, int], label_idx: int) -> Data:
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


def prepare_graph_datasets(spr: DatasetDict):
    token2idx = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(labels)}

    def _convert(split):
        graphs = []
        for seq, lab in zip(spr[split]["sequence"], spr[split]["label"]):
            graphs.append(sequence_to_graph(seq, token2idx, label2idx[lab]))
        return graphs

    return (
        {split: _convert(split) for split in ["train", "dev", "test"]},
        token2idx,
        label2idx,
    )


# ----------------- model -----------------
class GraphClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ----------------- load or create dataset -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
print(f"Looking for data at: {DATA_PATH}")
if DATA_PATH.exists():
    spr_bench = load_spr_bench(DATA_PATH)
else:
    print("Real dataset not found.  Using synthetic placeholder.")

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labs = [], []
        for i in range(n):
            L = random.randint(4, 8)
            tokens = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seq = " ".join(tokens)
            labs.append(random.choice(["yes", "no"]))
            seqs.append(seq)
        from datasets import Dataset

        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    from datasets import Dataset

    spr_bench = DatasetDict(
        {
            "train": Dataset.from_dict(synth(200)),
            "dev": Dataset.from_dict(synth(50)),
            "test": Dataset.from_dict(synth(50)),
        }
    )

# ----------------- preprocessing -----------------
graph_sets, token2idx, label2idx = prepare_graph_datasets(spr_bench)
num_classes = len(label2idx)
train_loader = DataLoader(graph_sets["train"], batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_sets["dev"], batch_size=128, shuffle=False)
test_loader = DataLoader(graph_sets["test"], batch_size=128, shuffle=False)

# ----------------- weight-decay sweep -----------------
weight_decay_values = [0, 1e-4, 5e-4, 1e-3, 1e-2]
EPOCHS = 5

for wd in weight_decay_values:
    run_key = f"weight_decay_{wd}"
    experiment_data[run_key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
    # fresh model & optimiser
    model = GraphClassifier(len(token2idx), 32, 64, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    print(f"\n=== Training run: weight_decay={wd} ===")
    # ----------------- training loop -----------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch.num_graphs
        train_loss = tot_loss / len(graph_sets["train"])
        experiment_data[run_key]["SPR_BENCH"]["losses"]["train"].append(train_loss)

        # ------ dev eval ------
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(dev_loader):
                seq_batch = spr_bench["dev"]["sequence"][
                    idx * 128 : idx * 128 + batch.num_graphs
                ]
                batch = batch.to(device)
                logits = model(batch)
                loss = cross_entropy(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())
                all_seqs.extend(seq_batch)
        val_loss /= len(graph_sets["dev"])
        experiment_data[run_key]["SPR_BENCH"]["losses"]["val"].append(val_loss)

        inv_label = {v: k for k, v in label2idx.items()}
        pred_lbls = [inv_label[p] for p in all_preds]
        true_lbls = [inv_label[t] for t in all_labels]
        cwa = color_weighted_accuracy(all_seqs, true_lbls, pred_lbls)
        swa = shape_weighted_accuracy(all_seqs, true_lbls, pred_lbls)
        hwa = harmonic_weighted_accuracy(cwa, swa)
        experiment_data[run_key]["SPR_BENCH"]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "hwa": hwa}
        )
        experiment_data[run_key]["SPR_BENCH"]["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}"
        )

    # ----------------- test evaluation -----------------
    model.eval()
    test_preds, test_labels, test_seqs = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            seq_batch = spr_bench["test"]["sequence"][
                idx * 128 : idx * 128 + batch.num_graphs
            ]
            batch = batch.to(device)
            logits = model(batch)
            test_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            test_labels.extend(batch.y.cpu().tolist())
            test_seqs.extend(seq_batch)
    inv_label = {v: k for k, v in label2idx.items()}
    pred_lbls = [inv_label[p] for p in test_preds]
    true_lbls = [inv_label[t] for t in test_labels]
    cwa_t = color_weighted_accuracy(test_seqs, true_lbls, pred_lbls)
    swa_t = shape_weighted_accuracy(test_seqs, true_lbls, pred_lbls)
    hwa_t = harmonic_weighted_accuracy(cwa_t, swa_t)
    print(f"Test results (wd={wd}): CWA={cwa_t:.3f} SWA={swa_t:.3f} HWA={hwa_t:.3f}")
    experiment_data[run_key]["SPR_BENCH"]["metrics"]["test"] = {
        "cwa": cwa_t,
        "swa": swa_t,
        "hwa": hwa_t,
    }
    experiment_data[run_key]["SPR_BENCH"]["predictions"] = pred_lbls
    experiment_data[run_key]["SPR_BENCH"]["ground_truth"] = true_lbls

# ----------------- save all results -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
