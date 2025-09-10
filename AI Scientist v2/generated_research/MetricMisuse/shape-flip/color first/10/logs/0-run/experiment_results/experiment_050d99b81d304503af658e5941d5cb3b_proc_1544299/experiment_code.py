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

# ----------------- experiment data container -----------------
experiment_data = {
    "batch_size": {  # hyper-parameter tuning type
        "SPR_BENCH": {}  # individual runs will be stored here keyed by batch size
    }
}

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
    if n == 1:  # self loop for single node graphs
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
        x = self.embed(data.x.squeeze())  # [N] -> [N,emb]
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)  # [batch,hidden]
        return self.lin(x)


# ----------------- attempt to load real dataset -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
print(f"Looking for data at: {DATA_PATH}")
if DATA_PATH.exists():
    spr_bench = load_spr_bench(DATA_PATH)
else:
    print("Real dataset not found.  Using synthetic placeholder.")

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labs = [], []
        for _ in range(n):
            L = random.randint(4, 8)
            tokens = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
            seqs.append(" ".join(tokens))
            labs.append(random.choice(["yes", "no"]))
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    from datasets import Dataset

    spr_bench = DatasetDict(
        {
            "train": Dataset.from_dict(synth(200)),
            "dev": Dataset.from_dict(synth(50)),
            "test": Dataset.from_dict(synth(50)),
        }
    )

# ----------------- build graph datasets -----------------
graph_sets, token2idx, label2idx = prepare_graph_datasets(spr_bench)
num_classes = len(label2idx)
print(f"Vocab size: {len(token2idx)} | #classes: {num_classes}")

# ----------------- hyper-parameter tuning loop -----------------
candidate_batch_sizes = [32, 64, 128]
EPOCHS = 5

for bs in candidate_batch_sizes:
    print(f"\n=== Training with batch size {bs} ===")
    run_key = str(bs)
    experiment_data["batch_size"]["SPR_BENCH"][run_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    # loaders
    train_loader = DataLoader(graph_sets["train"], batch_size=bs, shuffle=True)
    dev_loader = DataLoader(graph_sets["dev"], batch_size=128)
    test_loader = DataLoader(graph_sets["test"], batch_size=128)

    # model & optimiser
    model = GraphClassifier(
        vocab_size=len(token2idx), emb_dim=32, hidden_dim=64, num_classes=num_classes
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # ------------- training epochs -------------
    for epoch in range(1, EPOCHS + 1):
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
        experiment_data["batch_size"]["SPR_BENCH"][run_key]["losses"]["train"].append(
            train_loss
        )
        experiment_data["batch_size"]["SPR_BENCH"][run_key]["metrics"]["train"].append(
            {}
        )  # placeholder to respect structure

        # ---- evaluation on dev ----
        model.eval()
        all_preds, all_labels, all_seqs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dev_loader):
                seq_batch = spr_bench["dev"]["sequence"][
                    batch_idx * 128 : batch_idx * 128 + batch.num_graphs
                ]
                batch = batch.to(device)
                logits = model(batch)
                loss = cross_entropy(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=-1).cpu().tolist()
                labels = batch.y.cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_seqs.extend(seq_batch)
        val_loss /= len(graph_sets["dev"])
        experiment_data["batch_size"]["SPR_BENCH"][run_key]["losses"]["val"].append(
            val_loss
        )
        inv_label = {v: k for k, v in label2idx.items()}
        pred_lbls = [inv_label[p] for p in all_preds]
        true_lbls = [inv_label[t] for t in all_labels]
        cwa = color_weighted_accuracy(all_seqs, true_lbls, pred_lbls)
        swa = shape_weighted_accuracy(all_seqs, true_lbls, pred_lbls)
        hwa = harmonic_weighted_accuracy(cwa, swa)
        experiment_data["batch_size"]["SPR_BENCH"][run_key]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "hwa": hwa}
        )
        experiment_data["batch_size"]["SPR_BENCH"][run_key]["timestamps"].append(
            time.time()
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}"
        )

    # ------------- final test evaluation -------------
    model.eval()
    test_preds, test_labels, test_seqs = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            seq_batch = spr_bench["test"]["sequence"][
                batch_idx * 128 : batch_idx * 128 + batch.num_graphs
            ]
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch.y.cpu().tolist()
            test_preds.extend(preds)
            test_labels.extend(labels)
            test_seqs.extend(seq_batch)
    inv_label = {v: k for k, v in label2idx.items()}
    pred_lbls = [inv_label[p] for p in test_preds]
    true_lbls = [inv_label[t] for t in test_labels]
    cwa_test = color_weighted_accuracy(test_seqs, true_lbls, pred_lbls)
    swa_test = shape_weighted_accuracy(test_seqs, true_lbls, pred_lbls)
    hwa_test = harmonic_weighted_accuracy(cwa_test, swa_test)
    print(
        f"Test results (bs={bs})  CWA={cwa_test:.3f}  SWA={swa_test:.3f}  HWA={hwa_test:.3f}"
    )
    experiment_data["batch_size"]["SPR_BENCH"][run_key]["metrics"]["test"] = {
        "cwa": cwa_test,
        "swa": swa_test,
        "hwa": hwa_test,
    }
    experiment_data["batch_size"]["SPR_BENCH"][run_key]["predictions"] = pred_lbls
    experiment_data["batch_size"]["SPR_BENCH"][run_key]["ground_truth"] = true_lbls

# ----------------- save experiment data -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f'Experiment data saved to {os.path.join(working_dir, "experiment_data.npy")}')
