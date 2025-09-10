import os, pathlib, random, copy, time, numpy as np, torch, torch.nn as nn
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---------------- mandatory boiler-plate ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -------------------------------------------------------------------------


# ---------- try to load official SPR_BENCH via helper script -------------
def load_spr() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import SPR

        data_path = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(data_path)
        print("Loaded real SPR_BENCH")
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        print("Real SPR_BENCH missing, falling back to synthetic.")
        raise RuntimeError from e


def build_synthetic(n_train=10000, n_val=2000, n_test=2000):
    shapes, colors, labels = ["C", "S", "T"], ["r", "g", "b", "y"], ["rule1", "rule2"]

    def make_seq():
        L = random.randint(4, 12)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def make_split(n):
        return [
            {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
            for i in range(n)
        ]

    return make_split(n_train), make_split(n_val), make_split(n_test)


try:
    train_rows, dev_rows, test_rows = load_spr()
except RuntimeError:
    train_rows, dev_rows, test_rows = build_synthetic()


# ---------------- vocabulary & label mapping -----------------------------
def all_tokens(rows):
    for r in rows:
        for tok in r["sequence"].split():
            yield tok


token2idx = {"<PAD>": 0}
for tok in all_tokens(train_rows + dev_rows + test_rows):
    token2idx.setdefault(tok, len(token2idx))
label2idx = {}
for r in train_rows + dev_rows + test_rows:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print(f"Vocab size={len(token2idx)}  num_classes={num_classes}")


# ---------------- metrics -------------------------------------------------
def count_colors(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shapes(seq):
    return len({tok[0] for tok in seq.split() if tok})


def cmp_weight(seq):
    return count_colors(seq) + count_shapes(seq)


def weighted_acc(seqs, y_true, y_pred, weight_fn):
    w = [weight_fn(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc(seqs, y_true, y_pred, count_colors)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc(seqs, y_true, y_pred, count_shapes)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc(seqs, y_true, y_pred, cmp_weight)


# ---------------- graph construction ------------------------------------
def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    shapes = [t[0] for t in toks]
    colors = [t[1] if len(t) > 1 else "_" for t in toks]
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    src, dst, rel = [], [], []
    # relation 0: sequential neighbours
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        rel.extend([0, 0])
    # relation 1: same shape
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i] == shapes[j]:
                src.extend([i, j])
                dst.extend([j, i])
                rel.extend([1, 1])
    # relation 2: same color
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j]:
                src.extend([i, j])
                dst.extend([j, i])
                rel.extend([2, 2])
    # relation 3: position parity (even/odd)
    for i in range(n):
        for j in range(i + 1, n):
            if (i % 2) == (j % 2):
                src.extend([i, j])
                dst.extend([j, i])
                rel.extend([3, 3])
    if not src:
        src = [0]
        dst = [0]
        rel = [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)
    return Data(
        x=node_feats,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label2idx[label]], dtype=torch.long),
        seq=seq,
    )


train_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in test_rows]
batch_size = 256
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ---------------- model ---------------------------------------------------
class DeepRGCN(nn.Module):
    def __init__(
        self,
        vocab,
        embed_dim=128,
        hidden=128,
        num_rel=4,
        layers=4,
        n_cls=2,
        dropout=0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(embed_dim, hidden, num_relations=num_rel))
        for _ in range(layers - 2):
            self.convs.append(RGCNConv(hidden, hidden, num_relations=num_rel))
        self.convs.append(RGCNConv(hidden, hidden, num_relations=num_rel))
        self.lin = nn.Linear(hidden, n_cls)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.embed(x)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index, edge_type))
            x = self.dropout(x)
        g = global_mean_pool(x, batch)
        return self.lin(g)


# ---------------- training utils -----------------------------------------
def epoch_pass(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    tot_loss, seqs, ys, preds = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch.num_graphs
        pr = out.argmax(-1).cpu().tolist()
        gt = batch.y.squeeze().cpu().tolist()
        preds.extend(pr)
        ys.extend(gt)
        seqs.extend(batch.seq)
    loss_avg = tot_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, ys, preds)
    swa = shape_weighted_accuracy(seqs, ys, preds)
    cpx = complexity_weighted_accuracy(seqs, ys, preds)
    return loss_avg, cwa, swa, cpx, preds, ys, seqs


# ---------------- experiment tracking dict -------------------------------
experiment_data = {
    "SPR_DeepRGCN": {
        "metrics": {
            "train": {"CWA": [], "SWA": [], "CmpWA": []},
            "val": {"CWA": [], "SWA": [], "CmpWA": []},
        },
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- training loop ------------------------------------------
model = DeepRGCN(len(token2idx), n_cls=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_val = float("inf")
best_state = None
patience = 10
wait = 0
max_epochs = 100
start_time = time.time()
for epoch in range(1, max_epochs + 1):
    tr_l, tr_cwa, tr_swa, tr_cpx, _, _, _ = epoch_pass(
        model, train_loader, criterion, optimizer
    )
    val_l, val_cwa, val_swa, val_cpx, _, _, _ = epoch_pass(model, val_loader, criterion)
    scheduler.step()
    # log
    experiment_data["SPR_DeepRGCN"]["losses"]["train"].append(tr_l)
    experiment_data["SPR_DeepRGCN"]["losses"]["val"].append(val_l)
    experiment_data["SPR_DeepRGCN"]["metrics"]["train"]["CWA"].append(tr_cwa)
    experiment_data["SPR_DeepRGCN"]["metrics"]["train"]["SWA"].append(tr_swa)
    experiment_data["SPR_DeepRGCN"]["metrics"]["train"]["CmpWA"].append(tr_cpx)
    experiment_data["SPR_DeepRGCN"]["metrics"]["val"]["CWA"].append(val_cwa)
    experiment_data["SPR_DeepRGCN"]["metrics"]["val"]["SWA"].append(val_swa)
    experiment_data["SPR_DeepRGCN"]["metrics"]["val"]["CmpWA"].append(val_cpx)
    experiment_data["SPR_DeepRGCN"]["epochs"].append(epoch)
    print(f"Epoch {epoch:03d}: val_loss={val_l:.4f} CmpWA={val_cpx:.4f}")
    # early stopping
    if val_l < best_val - 1e-4:
        best_val = val_l
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------------- test evaluation ----------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
test_l, test_cwa, test_swa, test_cpx, preds, gt, _ = epoch_pass(
    model, test_loader, criterion
)
print(
    f"TEST: loss={test_l:.4f} CWA={test_cwa:.4f} SWA={test_swa:.4f} CmpWA={test_cpx:.4f}"
)
experiment_data["SPR_DeepRGCN"]["predictions"] = preds
experiment_data["SPR_DeepRGCN"]["ground_truth"] = gt
experiment_data["SPR_DeepRGCN"]["test_metrics"] = {
    "loss": test_l,
    "CWA": test_cwa,
    "SWA": test_swa,
    "CmpWA": test_cpx,
}

# ---------------- save ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to 'working/experiment_data.npy'")
print(f"Elapsed {time.time()-start_time:.1f}s")
