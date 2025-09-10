# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from datetime import datetime
from typing import List

# ------------------------- dirs & device ------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------- helper metrics -----------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_t: List[int], y_p: List[int]
) -> float:
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    return sum(w_i if t == p else 0 for w_i, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


# ------------------------- load (real or synth) SPR -------------------------
spr_root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
if spr_root.exists():
    from datasets import load_dataset

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(spr_root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dsets = {sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]}
    print("Loaded real SPR_BENCH.")
else:  # ---------- synthetic fallback
    print("SPR_BENCH not found, generating synthetic data.")

    def make_synth(n):
        shapes, colors = ["A", "B", "C"], ["1", "2", "3"]
        seqs, labels = [], []
        for _ in range(n):
            length = np.random.randint(4, 8)
            seqs.append(
                " ".join(
                    np.random.choice(shapes) + np.random.choice(colors)
                    for _ in range(length)
                )
            )
            labels.append(np.random.randint(0, 3))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    dsets = {"train": make_synth(500), "dev": make_synth(100), "test": make_synth(100)}

# ------------------------- vocab & graph builder ----------------------------
all_shapes, all_colors = set(), set()
for s in dsets["train"]["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0])
        all_colors.add(tok[1])
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
label2idx = {l: i for i, l in enumerate(sorted(set(dsets["train"]["label"])))}
num_shapes, num_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)


def seq_to_graph(seq: str, label: int):
    toks = seq.strip().split()
    n = len(toks)
    x = torch.tensor(
        np.stack([[shape2idx[t[0]], color2idx[t[1]]] for t in toks]), dtype=torch.long
    )
    if n > 1:
        src, dst = np.arange(n - 1), np.arange(1, n)
        edge_index = torch.tensor(
            np.vstack([np.hstack([src, dst]), np.hstack([dst, src])]), dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, seq=seq)


def build_dataset(split):
    return (
        [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]
        if isinstance(split, dict)
        else [seq_to_graph(rec["sequence"], rec["label"]) for rec in split]
    )


train_data, dev_data, test_data = map(
    build_dataset, (dsets["train"], dsets["dev"], dsets["test"])
)


# ------------------------- model --------------------------------------------
class SPRGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_emb, self.color_emb = nn.Embedding(num_shapes, 8), nn.Embedding(
            num_colors, 8
        )
        self.lin = nn.Linear(16, 32)
        self.conv1, self.conv2 = GraphConv(32, 64), GraphConv(64, 64)
        self.cls = nn.Linear(64, num_classes)

    def forward(self, data):
        x = torch.cat([self.shape_emb(data.x[:, 0]), self.color_emb(data.x[:, 1])], 1)
        x = F.relu(self.lin(x))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.cls(x)


# ------------------------- experiment dict ----------------------------------
experiment_data = {"batch_size": {}}  # <- hyperparam tuning type key
batch_sweep = [16, 32, 64, 128]
epochs = 5

for bs in batch_sweep:
    key = f"bs_{bs}"
    experiment_data["batch_size"][key] = {
        "metrics": {"train_compwa": [], "val_compwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    # loaders and model/opt
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=128, shuffle=False)
    model = SPRGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ------------- training loop -------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * batch.num_graphs
        avg_train_loss = tot_loss / len(train_loader.dataset)

        # validation
        model.eval()
        vloss = 0
        seqs = true = pred = []
        all_seq, all_true, all_pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                out = model(batch)
                l = criterion(out, batch.y)
                vloss += l.item() * batch.num_graphs
                pr = out.argmax(1).cpu().tolist()
                lb = batch.y.cpu().tolist()
                all_pred.extend(pr)
                all_true.extend(lb)
                all_seq.extend(batch.seq)
        avg_val_loss = vloss / len(dev_loader.dataset)
        compwa = complexity_weighted_accuracy(all_seq, all_true, all_pred)

        # log
        ed = experiment_data["batch_size"][key]
        ed["losses"]["train"].append(avg_train_loss)
        ed["losses"]["val"].append(avg_val_loss)
        ed["metrics"]["val_compwa"].append(compwa)
        ed["epochs"].append(epoch)
        print(
            f"[bs={bs}] Epoch {epoch}  train_loss {avg_train_loss:.4f}  "
            f"val_loss {avg_val_loss:.4f}  CompWA {compwa:.4f}"
        )

    # ------------- test ----------------------------------------------------
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            all_pred.extend(out.argmax(1).cpu().tolist())
            all_true.extend(batch.y.cpu().tolist())
            all_seq.extend(batch.seq)
    test_compwa = complexity_weighted_accuracy(all_seq, all_true, all_pred)
    experiment_data["batch_size"][key]["metrics"]["test_compwa"] = test_compwa
    experiment_data["batch_size"][key]["predictions"] = all_pred
    experiment_data["batch_size"][key]["ground_truth"] = all_true
    print(f"[bs={bs}] Test CompWA {test_compwa:.4f}")

    # ------------- plots ---------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure()
    plt.plot(
        experiment_data["batch_size"][key]["epochs"],
        experiment_data["batch_size"][key]["losses"]["train"],
        label="train",
    )
    plt.plot(
        experiment_data["batch_size"][key]["epochs"],
        experiment_data["batch_size"][key]["losses"]["val"],
        label="val",
    )
    plt.title(f"Loss (bs={bs})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_bs{bs}_{ts}.png"))
    plt.close()

    plt.figure()
    plt.plot(
        experiment_data["batch_size"][key]["epochs"],
        experiment_data["batch_size"][key]["metrics"]["val_compwa"],
    )
    plt.title(f"Val CompWA (bs={bs})")
    plt.xlabel("epoch")
    plt.ylabel("CompWA")
    plt.savefig(os.path.join(working_dir, f"compwa_bs{bs}_{ts}.png"))
    plt.close()

# ------------------------- save ---------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All runs complete. Data saved to ./working/experiment_data.npy")
