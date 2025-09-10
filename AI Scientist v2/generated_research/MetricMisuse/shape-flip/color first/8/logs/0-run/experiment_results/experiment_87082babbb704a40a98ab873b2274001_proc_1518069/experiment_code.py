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

import os, pathlib, numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------------- paths / device ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- load SPR-BENCH --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for sp in ["train", "dev", "test"]:
        dd[sp] = _load(f"{sp}.csv")
    return dd


DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists():  # create tiny synthetic fallback
    print("SPR_BENCH not found – creating tiny synthetic data.")
    shapes, colours = ["A", "B", "C"], ["1", "2", "3"]
    rng = np.random.default_rng(0)
    for split, sz in [("train", 200), ("dev", 40), ("test", 40)]:
        os.makedirs(DATA_PATH, exist_ok=True)
        fn = DATA_PATH / f"{split}.csv"
        with open(fn, "w") as f:
            f.write("id,sequence,label\n")
            for i in range(sz):
                n = rng.integers(3, 7)
                seq = " ".join(
                    rng.choice(shapes) + rng.choice(colours) for _ in range(n)
                )
                lbl = rng.choice(["yes", "no"])
                f.write(f"{split}_{i},{seq},{lbl}\n")

dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ------------------- helpers -------------------
def parse_token(tok):
    return tok[0], tok[1:] if len(tok) > 1 else "0"


# build full-token vocab (for the ablation)
all_tokens = set()
for row in dsets["train"]:
    all_tokens.update(row["sequence"].split())
token2id = {t: i for i, t in enumerate(sorted(all_tokens))}
print("Vocab size:", len(token2id))

# label mapping
all_labels = sorted({row["label"] for row in dsets["train"]})
label2id = {l: i for i, l in enumerate(all_labels)}


# ---------------- sequence  ➜  graph ----------------
def seq_to_graph(sequence, lbl):
    toks = sequence.split()
    ids = [token2id[t] for t in toks]
    x = torch.tensor(ids, dtype=torch.long)  # 1-D long tensor
    n = len(ids)
    if n > 1:
        src = torch.arange(0, n - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2id[lbl]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(split):
    return [seq_to_graph(r["sequence"], r["label"]) for r in dsets[split]]


graph_train = build_graph_dataset("train")
graph_dev = build_graph_dataset("dev")
graph_test = build_graph_dataset("test")

train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
dev_loader = DataLoader(graph_dev, batch_size=128, shuffle=False)
test_loader = DataLoader(graph_test, batch_size=128, shuffle=False)


# ---------------- model (token-level embedding) ----------------
class GCNTokenLevel(torch.nn.Module):
    def __init__(self, vocab, emb_dim=32, hid=64, num_classes=len(label2id)):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, emb_dim)
        self.conv1 = GCNConv(emb_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, num_classes)

    def forward(self, x_token, edge_index, batch):
        x = self.embed(x_token)  # (N, emb_dim) float
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = GCNTokenLevel(len(token2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------- complexity-weighted accuracy ----------------
def complexity_weight(seq):
    toks = seq.split()
    shapes = {t[0] for t in toks}
    cols = {t[1:] if len(t) > 1 else "0" for t in toks}
    return len(shapes) + len(cols)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) else 0.0


# ---------------- tracking dict ----------------
experiment_data = {
    "token_level": {  # ablation type
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------------- training ----------------
EPOCHS = 10
for ep in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tot_loss = tot_corr = tot_ex = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        tot_corr += int((out.argmax(-1) == batch.y).sum().item())
        tot_ex += batch.num_graphs
    tr_loss, tr_acc = tot_loss / tot_ex, tot_corr / tot_ex
    experiment_data["token_level"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["token_level"]["SPR_BENCH"]["metrics"]["train"].append(tr_acc)

    # ---- validation ----
    model.eval()
    v_loss = v_corr = v_ex = 0
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            v_loss += loss.item() * batch.num_graphs
            v_corr += int((out.argmax(-1) == batch.y).sum().item())
            v_ex += batch.num_graphs
    val_loss, val_acc = v_loss / v_ex, v_corr / v_ex
    experiment_data["token_level"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["token_level"]["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    print(f"Epoch {ep}: val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

# --------- final evaluation (dev) for CompWA ----------
seqs = [row["sequence"] for row in dsets["dev"]]
model.eval()
preds = []
with torch.no_grad():
    for batch in dev_loader:
        batch = batch.to(device)
        preds.extend(
            model(batch.x, batch.edge_index, batch.batch).argmax(-1).cpu().tolist()
        )

compwa = comp_weighted_accuracy(
    seqs, [label2id[r["label"]] for r in dsets["dev"]], preds
)
print(f"Complexity-Weighted Accuracy (dev): {compwa:.4f}")

experiment_data["token_level"]["SPR_BENCH"]["predictions"] = preds
experiment_data["token_level"]["SPR_BENCH"]["ground_truth"] = [
    label2id[r["label"]] for r in dsets["dev"]
]

# ---------------- save + plot ----------------
plt.figure()
plt.plot(experiment_data["token_level"]["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["token_level"]["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Cross-Entropy loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Data & plot saved to ./working")
