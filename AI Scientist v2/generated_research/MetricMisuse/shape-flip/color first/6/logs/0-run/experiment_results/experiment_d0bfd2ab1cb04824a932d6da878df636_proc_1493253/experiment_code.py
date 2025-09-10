import os, pathlib, random, math, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List

# ------------------- mandatory working dir ----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- mandatory device print ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- experiment data dict -----------------------------------
experiment_data = {
    "spr_bench": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------- helper: metrics ----------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    num = sum(w for w, t, p in zip(weights, y_true, y_pred) if t == p)
    den = max(sum(weights), 1)
    return num / den


# ------------------- dataset resolver ---------------------------------------
POSSIBLE_PATHS = [
    os.getenv("SPR_BENCH_PATH", ""),
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    "./SPR_BENCH",
]
spr_root = None
for p in POSSIBLE_PATHS:
    if p and pathlib.Path(p).expanduser().exists():
        spr_root = pathlib.Path(p).expanduser()
        break

if spr_root:
    print(f"Loading real SPR_BENCH from {spr_root}")

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(spr_root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    raw_dsets = {sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]}
else:
    print("Real SPR_BENCH not found – generating *rule-based* synthetic data.")

    shapes, colors = ["A", "B", "C", "D"], ["1", "2", "3", "4"]

    def gen_rule_based(n):
        seqs, labels = [], []
        for i in range(n):
            ln = random.randint(4, 8)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(ln)
            )
            # RULE: label is the index of the *most common shape* in the sequence
            shape_cnt = {s: seq.split().count(s + c) for s in shapes for c in colors}
            majority_shape = max(
                shapes, key=lambda s: sum(shape_cnt[s + c] for c in colors)
            )
            label = shapes.index(majority_shape)
            seqs.append(seq)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    raw_dsets = {
        "train": gen_rule_based(4000),
        "dev": gen_rule_based(1000),
        "test": gen_rule_based(1000),
    }

# ------------------- vocabulary ---------------------------------------------
all_shapes, all_colors = set(), set()
for s in raw_dsets["train"]["sequence"]:
    for tok in s.split():
        all_shapes.add(tok[0])
        all_colors.add(tok[1])
shape2idx = {s: i for i, s in enumerate(sorted(all_shapes))}
color2idx = {c: i for i, c in enumerate(sorted(all_colors))}
label_set = sorted(set(raw_dsets["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
num_shapes, num_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)


# ------------------- graph builder ------------------------------------------
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.strip().split()
    n = len(toks)
    x = torch.tensor(
        [[shape2idx[t[0]], color2idx[t[1]]] for t in toks], dtype=torch.long
    )
    edges_src, edges_dst, e_types = [], [], []

    # relation 0: sequential neighbours (bidirectional)
    for i in range(n - 1):
        edges_src += [i, i + 1]
        edges_dst += [i + 1, i]
        e_types += [0, 0]
    # relation 1: same color
    col_map = {}
    for idx, t in enumerate(toks):
        col_map.setdefault(t[1], []).append(idx)
    for nodes in col_map.values():
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)
                    e_types.append(1)
    # relation 2: same shape
    shp_map = {}
    for idx, t in enumerate(toks):
        shp_map.setdefault(t[0], []).append(idx)
    for nodes in shp_map.values():
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)
                    e_types.append(2)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(e_types, dtype=torch.long)
    y = torch.tensor([label2idx[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=seq)


def build_dataset(split):
    if isinstance(split, dict):  # synthetic dict
        seqs, labels = split["sequence"], split["label"]
    else:  # HF Dataset
        seqs, labels = split["sequence"], split["label"]
    return [seq_to_graph(s, l) for s, l in zip(seqs, labels)]


train_ds, dev_ds, test_ds = map(
    build_dataset, (raw_dsets["train"], raw_dsets["dev"], raw_dsets["test"])
)


# ------------------- model ---------------------------------------------------
class SPR_RGCN(nn.Module):
    def __init__(self, emb_dim=16, hidden=64, num_relations=3):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, emb_dim // 2)
        self.color_emb = nn.Embedding(num_colors, emb_dim // 2)
        self.lin0 = nn.Linear(emb_dim, hidden)
        self.rgcn1 = RGCNConv(hidden, hidden, num_relations)
        self.rgcn2 = RGCNConv(hidden, hidden, num_relations)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, data):
        s_e = self.shape_emb(data.x[:, 0])
        c_e = self.color_emb(data.x[:, 1])
        x = torch.cat([s_e, c_e], 1)
        x = F.relu(self.lin0(x))
        x = F.relu(self.rgcn1(x, data.edge_index, data.edge_type))
        x = F.relu(self.rgcn2(x, data.edge_index, data.edge_type))
        x = global_mean_pool(x, data.batch)
        return self.cls(x)


model = SPR_RGCN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------- loaders -------------------------------------------------
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

# ------------------- training ------------------------------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    avg_train_loss = train_loss / len(train_loader.dataset)

    # validation
    model.eval()
    val_loss, v_seqs, v_true, v_pred = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch)
            l = criterion(out, batch.y)
            val_loss += l.item() * batch.num_graphs
            preds = out.argmax(1).cpu().tolist()
            v_pred.extend(preds)
            v_true.extend(batch.y.cpu().tolist())
            v_seqs.extend(batch.seq)
    avg_val_loss = val_loss / len(dev_loader.dataset)
    val_cplx = complexity_weighted_accuracy(v_seqs, v_true, v_pred)

    print(
        f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}, Val CplxWA = {val_cplx:.4f}"
    )

    # log
    ed = experiment_data["spr_bench"]
    ed["losses"]["train"].append(avg_train_loss)
    ed["losses"]["val"].append(avg_val_loss)
    ed["metrics"]["val"].append(val_cplx)
    ed["epochs"].append(epoch)

# ------------------- testing -------------------------------------------------
model.eval()
t_seqs, t_true, t_pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        preds = out.argmax(1).cpu().tolist()
        t_pred.extend(preds)
        t_true.extend(batch.y.cpu().tolist())
        t_seqs.extend(batch.seq)
test_cplx = complexity_weighted_accuracy(t_seqs, t_true, t_pred)
print(f"Test Complexity-Weighted Accuracy: {test_cplx:.4f}")

experiment_data["spr_bench"]["metrics"]["test"] = test_cplx
experiment_data["spr_bench"]["predictions"] = t_pred
experiment_data["spr_bench"]["ground_truth"] = t_true

# ------------------- plots ---------------------------------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure()
plt.plot(
    experiment_data["spr_bench"]["epochs"],
    experiment_data["spr_bench"]["losses"]["train"],
    label="train",
)
plt.plot(
    experiment_data["spr_bench"]["epochs"],
    experiment_data["spr_bench"]["losses"]["val"],
    label="val",
)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss curve")
plt.savefig(os.path.join(working_dir, f"loss_{ts}.png"))
plt.close()

plt.figure()
plt.plot(
    experiment_data["spr_bench"]["epochs"],
    experiment_data["spr_bench"]["metrics"]["val"],
)
plt.xlabel("epoch")
plt.ylabel("CplxWA")
plt.title("Validation CplxWA")
plt.savefig(os.path.join(working_dir, f"cplxwa_{ts}.png"))
plt.close()

# ------------------- save artefacts -----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to ./working/experiment_data.npy")
