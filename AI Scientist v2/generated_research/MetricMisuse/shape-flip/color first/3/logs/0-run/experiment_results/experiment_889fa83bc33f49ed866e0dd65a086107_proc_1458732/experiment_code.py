import os, pathlib, time, copy, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import List, Dict

# -------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------- locate SPR_BENCH ----------------
def locate_spr_bench() -> pathlib.Path:
    cands = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in cands:
        if p and (p / "train.csv").exists():
            return p.resolve()
    raise FileNotFoundError("Place SPR_BENCH dataset or set SPR_DATA_PATH.")


DATA_PATH = locate_spr_bench()


# -------- helpers from starter ------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def count_color_variety(seq: str) -> int:
    return len(set(t[1] for t in seq.strip().split() if len(t) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.strip().split() if t))


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_t, y_p)]
    return sum(corr) / (sum(w) if sum(w) > 0 else 1)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_t, y_p)]
    return sum(corr) / (sum(w) if sum(w) > 0 else 1)


def structure_weighted_accuracy(seqs, y_t, y_p):
    def struct_cmplx(s):
        toks = s.strip().split()
        return len(set((tok[0], tok[1]) for tok in toks if len(tok) > 1))

    w = [struct_cmplx(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_t, y_p)]
    return sum(corr) / (sum(w) if sum(w) > 0 else 1)


# -------- load dataset --------------------
spr = load_spr_bench(DATA_PATH)


# vocab + label maps -----------------------
def tok_list(seq):
    return seq.strip().split()


token_set = set()
label_set = set()
for ex in spr["train"]:
    token_set.update(tok_list(ex["sequence"]))
    label_set.add(ex["label"])
token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
shape2idx = {ch: i + 1 for i, ch in enumerate(sorted({t[0] for t in token_set}))}
color2idx = {
    ch: i + 1 for i, ch in enumerate(sorted({t[1] for t in token_set if len(t) > 1}))
}
label2idx = {lab: i for i, lab in enumerate(sorted(label_set))}
idx2label = {i: lab for lab, i in label2idx.items()}


# -------- graph construction --------------
def seq_to_graph(example) -> Data:
    seq = example["sequence"]
    tokens = tok_list(seq)
    n = len(tokens)
    tok_idx = [token2idx[t] for t in tokens]
    shp_idx = [shape2idx[t[0]] for t in tokens]
    col_idx = [color2idx.get(t[1], 0) for t in tokens]
    # node feature indices
    x = torch.tensor(np.vstack([tok_idx, shp_idx, col_idx]).T, dtype=torch.long)
    # edges: sequence order
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    # same shape edges
    groups = {}
    for i, sid in enumerate(shp_idx):
        groups.setdefault(sid, []).append(i)
    for g in groups.values():
        for i in g:
            for j in g:
                if i != j:
                    edges.append((i, j))
    # same color edges
    groups = {}
    for i, cid in enumerate(col_idx):
        groups.setdefault(cid, []).append(i)
    for g in groups.values():
        for i in g:
            for j in g:
                if i != j:
                    edges.append((i, j))
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).T
    y = torch.tensor([label2idx[example["label"]]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.seq = seq
    return data


train_graphs = [seq_to_graph(ex) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex) for ex in spr["dev"]]
test_graphs = [seq_to_graph(ex) for ex in spr["test"]]

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=256, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=256, shuffle=False)


# -------- model ---------------------------
class SPRGraphSAGE(nn.Module):
    def __init__(self, vocab_sz, shape_sz, color_sz, num_classes, emb_dim=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_sz + 1, emb_dim, padding_idx=0)
        self.shape_emb = nn.Embedding(shape_sz + 1, emb_dim // 2, padding_idx=0)
        self.color_emb = nn.Embedding(color_sz + 1, emb_dim // 2, padding_idx=0)
        in_dim = emb_dim + emb_dim // 2 + emb_dim // 2
        self.conv1 = SAGEConv(in_dim, 64)
        self.conv2 = SAGEConv(64, 64)
        self.lin = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        tok, shp, col = x[:, 0], x[:, 1], x[:, 2]
        h = torch.cat(
            [self.tok_emb(tok), self.shape_emb(shp), self.color_emb(col)], dim=-1
        )
        h = self.conv1(h, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = global_mean_pool(h, batch)
        return self.lin(h)


model = SPRGraphSAGE(len(token2idx), len(shape2idx), len(color2idx), len(label2idx)).to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -------- storage dict --------------------
experiment_data = {
    "spr_bench": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -------- evaluation ----------------------
def run_eval(loader):
    model.eval()
    total_loss = 0
    y_true, y_pred, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(-1).cpu().tolist()
            labels = batch.y.view(-1).cpu().tolist()
            seqs.extend(batch.seq)
            y_true.extend(labels)
            y_pred.extend(preds)
    loss = total_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    strwa = structure_weighted_accuracy(seqs, y_true, y_pred)
    bwa = (cwa + swa + strwa) / 3.0
    return loss, bwa, cwa, swa, strwa, y_pred, y_true


# -------- training loop -------------------
best_val = -1
patience = 3
wait = 0
max_epochs = 30
for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    train_loss = epoch_loss / len(train_loader.dataset)
    _, train_bwa, _, _, _, _, _ = run_eval(train_loader)
    val_loss, val_bwa, val_cwa, val_swa, val_strwa, _, _ = run_eval(dev_loader)

    # log
    experiment_data["spr_bench"]["losses"]["train"].append(train_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)
    experiment_data["spr_bench"]["metrics"]["train"].append(train_bwa)
    experiment_data["spr_bench"]["metrics"]["val"].append(val_bwa)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"BWA={val_bwa:.4f}  (CWA={val_cwa:.4f} SWA={val_swa:.4f} StrWA={val_strwa:.4f})"
    )

    # early stop
    if val_bwa > best_val:
        best_val = val_bwa
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# -------- restore best model --------------
model.load_state_dict(best_state)

# -------- test evaluation -----------------
test_loss, test_bwa, test_cwa, test_swa, test_strwa, test_preds, test_labels = run_eval(
    test_loader
)
experiment_data["spr_bench"]["predictions"] = test_preds
experiment_data["spr_bench"]["ground_truth"] = test_labels
experiment_data["spr_bench"]["test_metrics"] = {
    "loss": test_loss,
    "BWA": test_bwa,
    "CWA": test_cwa,
    "SWA": test_swa,
    "StrWA": test_strwa,
}
print(
    f"Test -> loss={test_loss:.4f}  BWA={test_bwa:.4f} (CWA={test_cwa:.4f} "
    f"SWA={test_swa:.4f} StrWA={test_strwa:.4f})"
)

# -------- plot metric curves --------------
epochs = np.arange(1, len(experiment_data["spr_bench"]["metrics"]["train"]) + 1)
plt.figure()
plt.plot(epochs, experiment_data["spr_bench"]["metrics"]["train"], label="Train BWA")
plt.plot(epochs, experiment_data["spr_bench"]["metrics"]["val"], label="Dev BWA")
plt.xlabel("Epoch")
plt.ylabel("BWA")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(working_dir, "bwa_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"Curve saved to {plot_path}")

# -------- save numpy data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiment data saved.")
