# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from typing import List
from datetime import datetime

# ---- Device -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---- Helper metric functions ------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weighted_accuracy(
    sequences: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_color_variety(s) + count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    denom = sum(weights)
    return sum(correct) / denom if denom > 0 else 0.0


# ---- Try loading real SPR_BENCH  --------------------------------------------
def load_real_spr(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = {}
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
have_real_data = spr_root.exists()
if have_real_data:
    dsets = load_real_spr(spr_root)
    print("Loaded real SPR_BENCH.")
else:
    # ------ tiny synthetic fallback -----------------------------------------
    print("SPR_BENCH not found, creating synthetic toy data.")

    def make_synth(n):
        seqs, labels = [], []
        shapes = ["A", "B", "C"]
        colors = ["1", "2", "3"]
        for i in range(n):
            length = np.random.randint(4, 8)
            seq = " ".join(
                np.random.choice(shapes) + np.random.choice(colors)
                for _ in range(length)
            )
            seqs.append(seq)
            labels.append(np.random.randint(0, 3))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    dsets = {"train": make_synth(500), "dev": make_synth(100), "test": make_synth(100)}

# ---- Build vocabularies -----------------------------------------------------
all_shapes = set()
all_colors = set()
all_labels = set()
for ex in dsets["train"]["sequence"]:
    for tok in ex.split():
        if len(tok) >= 2:
            all_shapes.add(tok[0])
            all_colors.add(tok[1])
for lab in dsets["train"]["label"]:
    all_labels.add(lab)

shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
num_shapes = len(shape2idx)
num_colors = len(color2idx)
label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
num_classes = len(label2idx)


def seq_to_graph(seq: str, label: int):
    toks = seq.strip().split()
    n = len(toks)
    shape_ids = [shape2idx[t[0]] for t in toks]
    color_ids = [color2idx[t[1]] for t in toks]
    x = torch.tensor(np.stack([shape_ids, color_ids], 1), dtype=torch.long)
    # edges: consecutive tokens, bidirectional
    if n > 1:
        src = np.arange(n - 1)
        dst = np.arange(1, n)
        edge_index = torch.tensor(
            np.vstack([np.hstack([src, dst]), np.hstack([dst, src])]), dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def build_dataset(split_dict):
    if isinstance(split_dict, dict):  # synthetic fallback
        return [
            seq_to_graph(s, l)
            for s, l in zip(split_dict["sequence"], split_dict["label"])
        ]
    else:  # HuggingFace dataset
        return [seq_to_graph(rec["sequence"], rec["label"]) for rec in split_dict]


train_data = build_dataset(dsets["train"])
dev_data = build_dataset(dsets["dev"])
test_data = build_dataset(dsets["test"])


# ---- Model ------------------------------------------------------------------
class SPRGNN(nn.Module):
    def __init__(self, num_shapes, num_colors, num_classes):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, 8)
        self.color_emb = nn.Embedding(num_colors, 8)
        self.lin_node = nn.Linear(16, 32)
        self.conv1 = GraphConv(32, 64)
        self.conv2 = GraphConv(64, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, data):
        shape_e = self.shape_emb(data.x[:, 0])
        color_e = self.color_emb(data.x[:, 1])
        x = torch.cat([shape_e, color_e], dim=1)
        x = F.relu(self.lin_node(x))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)


model = SPRGNN(num_shapes, num_colors, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---- DataLoaders ------------------------------------------------------------
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=128, shuffle=False)

# ---- Experiment data dict ---------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_compwa": [], "val_compwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---- Training loop ----------------------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    avg_train_loss = total_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(avg_train_loss)

    # ---- Validation ---------------------------------------------------------
    model.eval()
    val_loss = 0
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            val_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1).cpu().tolist()
            labels = batch.y.cpu().tolist()
            seqs = batch.seq
            all_seq.extend(seqs)
            all_true.extend(labels)
            all_pred.extend(preds)
    avg_val_loss = val_loss / len(dev_loader.dataset)
    compwa = complexity_weighted_accuracy(all_seq, all_true, all_pred)

    experiment_data["SPR_BENCH"]["losses"]["val"].append(avg_val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_compwa"].append(compwa)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss = {avg_train_loss:.4f} | "
        f"val_loss = {avg_val_loss:.4f} | CompWA = {compwa:.4f}"
    )

# ---- Final evaluation on test set ------------------------------------------
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
model.eval()
all_seq, all_true, all_pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds = out.argmax(dim=1).cpu().tolist()
        labels = batch.y.cpu().tolist()
        all_seq.extend(batch.seq)
        all_true.extend(labels)
        all_pred.extend(preds)
test_compwa = complexity_weighted_accuracy(all_seq, all_true, all_pred)
print(f"Test CompWA: {test_compwa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = all_pred
experiment_data["SPR_BENCH"]["ground_truth"] = all_true
experiment_data["SPR_BENCH"]["metrics"]["test_compwa"] = test_compwa

# ---- Save metrics -----------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---- Plot losses and CompWA -------------------------------------------------
plt.figure()
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["train"],
    label="train_loss",
)
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["val"],
    label="val_loss",
)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss trajectory")
plt.savefig(os.path.join(working_dir, f"loss_{timestamp}.png"))

plt.figure()
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["metrics"]["val_compwa"],
    label="Val CompWA",
)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("CompWA")
plt.title("Validation Complexity-Weighted Accuracy")
plt.savefig(os.path.join(working_dir, f"compwa_{timestamp}.png"))
print("Training complete. Figures and data saved in ./working")
