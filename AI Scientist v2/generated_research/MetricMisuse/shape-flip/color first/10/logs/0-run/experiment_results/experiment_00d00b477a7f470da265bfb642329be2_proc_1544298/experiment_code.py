import os, pathlib, random, time, math, sys, itertools, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from datasets import DatasetDict
from typing import List, Dict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ----------------- experiment dict -----------------
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "lr_values": [],
            "metrics": {"train": [], "val": [], "test": None},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

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


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {tok for seq in seqs for tok in seq.split()}
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
        return [
            sequence_to_graph(s, token2idx, label2idx[l])
            for s, l in zip(spr[split]["sequence"], spr[split]["label"])
        ]

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


# ----------------- get dataset (real or synthetic) -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr_bench = load_spr_bench(DATA_PATH)
else:
    print("Real dataset not found. Using synthetic placeholder.")

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labs = [], []
        for _ in range(n):
            L = random.randint(4, 8)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            seqs.append(seq)
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

graph_sets, token2idx, label2idx = prepare_graph_datasets(spr_bench)
num_classes = len(label2idx)
print(f"Vocab size: {len(token2idx)} | #classes: {num_classes}")

train_loader_all = DataLoader(graph_sets["train"], batch_size=64, shuffle=True)
dev_loader_all = DataLoader(graph_sets["dev"], batch_size=128)
test_loader_all = DataLoader(graph_sets["test"], batch_size=128)

inv_label = {v: k for k, v in label2idx.items()}


# ----------------- training util -----------------
def run_training(lr, epochs=5):
    model = GraphClassifier(len(token2idx), 32, 64, num_classes).to(device)
    opt = Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_metrics = [], [], []
    for ep in range(1, epochs + 1):
        model.train()
        tloss = 0.0
        for batch in train_loader_all:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            tloss += loss.item() * batch.num_graphs
        tloss /= len(graph_sets["train"])
        train_losses.append(tloss)

        # val
        model.eval()
        vloss = 0.0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for idx, batch in enumerate(dev_loader_all):
                seq_batch = spr_bench["dev"]["sequence"][
                    idx * 128 : idx * 128 + batch.num_graphs
                ]
                batch = batch.to(device)
                outs = model(batch)
                loss = cross_entropy(outs, batch.y)
                vloss += loss.item() * batch.num_graphs
                preds += outs.argmax(-1).cpu().tolist()
                labels += batch.y.cpu().tolist()
                seqs += seq_batch
        vloss /= len(graph_sets["dev"])
        val_losses.append(vloss)
        pred_lbl = [inv_label[p] for p in preds]
        true_lbl = [inv_label[t] for t in labels]
        cwa = color_weighted_accuracy(seqs, true_lbl, pred_lbl)
        swa = shape_weighted_accuracy(seqs, true_lbl, pred_lbl)
        hwa = harmonic_weighted_accuracy(cwa, swa)
        val_metrics.append({"cwa": cwa, "swa": swa, "hwa": hwa})
        print(
            f"[lr={lr}] Epoch {ep}: train_loss={tloss:.4f} val_loss={vloss:.4f} HWA={hwa:.3f}"
        )
    return model, train_losses, val_losses, val_metrics


# ----------------- learning-rate sweep -----------------
lr_values = [3e-4, 5e-4, 1e-3, 2e-3]
best_hwa, best_lr, best_model = -1, None, None

for lr in lr_values:
    model, tr_losses, va_losses, va_metrics = run_training(lr, epochs=5)
    experiment_data["learning_rate"]["SPR_BENCH"]["lr_values"].append(lr)
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["train"].append(tr_losses)
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["val"].append(va_losses)
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["val"].append(va_metrics)
    experiment_data["learning_rate"]["SPR_BENCH"]["timestamps"].append(time.time())
    final_hwa = va_metrics[-1]["hwa"]
    if final_hwa > best_hwa:
        best_hwa, best_lr, best_model = final_hwa, lr, model

print(f"Best lr according to dev HWA: {best_lr} (HWA={best_hwa:.3f})")

# ----------------- test evaluation with best model -----------------
best_model.eval()
preds_test = []
labels_test = []
seqs_test = []
with torch.no_grad():
    for idx, batch in enumerate(test_loader_all):
        seq_batch = spr_bench["test"]["sequence"][
            idx * 128 : idx * 128 + batch.num_graphs
        ]
        batch = batch.to(device)
        outs = best_model(batch)
        preds_test += outs.argmax(-1).cpu().tolist()
        labels_test += batch.y.cpu().tolist()
        seqs_test += seq_batch
pred_lbl = [inv_label[p] for p in preds_test]
true_lbl = [inv_label[t] for t in labels_test]
cwa_test = color_weighted_accuracy(seqs_test, true_lbl, pred_lbl)
swa_test = shape_weighted_accuracy(seqs_test, true_lbl, pred_lbl)
hwa_test = harmonic_weighted_accuracy(cwa_test, swa_test)
print(f"Test  CWA={cwa_test:.3f}  SWA={swa_test:.3f}  HWA={hwa_test:.3f}")

experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["test"] = {
    "cwa": cwa_test,
    "swa": swa_test,
    "hwa": hwa_test,
}
experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["train"] = "N/A"
experiment_data["learning_rate"]["SPR_BENCH"]["predictions"] = pred_lbl
experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"] = true_lbl
experiment_data["learning_rate"]["SPR_BENCH"]["best_lr"] = best_lr

# ----------------- save -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Data saved to {os.path.join(working_dir,"experiment_data.npy")}')
