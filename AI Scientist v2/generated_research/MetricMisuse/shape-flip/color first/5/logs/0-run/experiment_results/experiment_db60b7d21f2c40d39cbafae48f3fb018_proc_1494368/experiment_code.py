import os, pathlib, random, time, copy, numpy as np, torch, torch.nn as nn
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# -------------------------------------------------------------------------
# boiler-plate: working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------------
# try real dataset, else fall back to tiny synthetic one
def load_spr() -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import SPR, importlib

        DATA_PATH = pathlib.Path(os.environ.get("SPR_DATA_PATH", "./SPR_BENCH"))
        dset = SPR.load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH.")
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        print("Could not load real dataset – using synthetic data.")
        shapes, colors, labels = (
            ["C", "S", "T"],
            ["r", "g", "b", "y"],
            ["rule1", "rule2"],
        )

        def make_seq():
            return " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 8))
            )

        def make_split(n):
            return [
                {"id": i, "sequence": make_seq(), "label": random.choice(labels)}
                for i in range(n)
            ]

        return make_split(600), make_split(200), make_split(200)


train_rows, dev_rows, test_rows = load_spr()


# -------------------------------------------------------------------------
# vocab + label maps
def extract_tokens(rows):
    for r in rows:
        for t in r["sequence"].split():
            yield t


token2idx = {"<PAD>": 0}
for tok in extract_tokens(train_rows + dev_rows + test_rows):
    token2idx.setdefault(tok, len(token2idx))
label2idx = {}
for r in train_rows + dev_rows + test_rows:
    label2idx.setdefault(r["label"], len(label2idx))
num_classes = len(label2idx)
print(f"Vocab={len(token2idx)}, Classes={num_classes}")


# -------------------------------------------------------------------------
# helpers for metrics
def count_color_variety(seq):
    return len({t[1] for t in seq.split()})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split()})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# -------------------------------------------------------------------------
# build graph with order / color / shape edges
def seq_to_graph(seq: str, label: str) -> Data:
    toks = seq.split()
    n = len(toks)
    node_feats = torch.tensor([token2idx[t] for t in toks], dtype=torch.long)
    src, dst = [], []
    # sequential edges
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    # color & shape edges
    color2idx, shape2idx = {}, {}
    for i, t in enumerate(toks):
        color2idx.setdefault(t[1], []).append(i)
        shape2idx.setdefault(t[0], []).append(i)
    for idxs in list(color2idx.values()) + list(shape2idx.values()):
        for i in idxs:
            for j in idxs:
                if i != j:
                    src.append(i)
                    dst.append(j)
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.zeros(2, 0, dtype=torch.long)
    )
    return Data(
        x=node_feats,
        edge_index=edge_index,
        y=torch.tensor([label2idx[label]], dtype=torch.long),
        seq=seq,
    )


train_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in train_rows]
val_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in dev_rows]
test_graphs = [seq_to_graph(r["sequence"], r["label"]) for r in test_rows]
batch_size = 64
train_loader, val_loader, test_loader = (
    DataLoader(g, batch_size=batch_size, shuffle=s)
    for g, s in [(train_graphs, True), (val_graphs, False), (test_graphs, False)]
)


# -------------------------------------------------------------------------
# model
class SPR_GNN(nn.Module):
    def __init__(self, vocab, embed=64, hidden=64, classes=2, drop=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=0)
        self.conv1, self.conv2 = GCNConv(embed, hidden), GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, classes)
        self.drop = nn.Dropout(drop)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# -------------------------------------------------------------------------
def run_epoch(model, loader, crit, opt=None):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    total_loss, seqs, y_true, y_pred = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = crit(out, batch.y)
        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(-1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(batch.y.cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cmp = complexity_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, (cwa, swa, cmp), y_pred, y_true, seqs


# -------------------------------------------------------------------------
max_epochs = 40
patience = 7
model = SPR_GNN(len(token2idx), classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
experiment_data = {
    "SPR": {
        "metrics": {
            "train": {"cwa": [], "swa": [], "cmp": []},
            "val": {"cwa": [], "swa": [], "cmp": []},
        },
        "losses": {"train": [], "val": []},
        "epochs": [],
    }
}
best_val = float("inf")
best_state = None
wait = 0
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_m, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_m, _, _, _ = run_epoch(model, val_loader, criterion)
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    for k, (tr_v, val_v) in zip(["cwa", "swa", "cmp"], zip(tr_m, val_m)):
        experiment_data["SPR"]["metrics"]["train"][k].append(tr_v)
        experiment_data["SPR"]["metrics"]["val"][k].append(val_v)
    experiment_data["SPR"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, "
        f"CWA={val_m[0]:.3f}, SWA={val_m[1]:.3f}, CmpWA={val_m[2]:.3f}"
    )
    if val_loss < best_val - 1e-4:
        best_val, wait = val_loss, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_m, test_pred, test_true, test_seq = run_epoch(
    model, test_loader, criterion
)
print(
    f"\nTEST  loss={test_loss:.4f}  CWA={test_m[0]:.3f}  SWA={test_m[1]:.3f}  CmpWA={test_m[2]:.3f}"
)

experiment_data["SPR"]["test"] = {
    "loss": test_loss,
    "cwa": test_m[0],
    "swa": test_m[1],
    "cmp": test_m[2],
    "pred": test_pred,
    "true": test_true,
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
