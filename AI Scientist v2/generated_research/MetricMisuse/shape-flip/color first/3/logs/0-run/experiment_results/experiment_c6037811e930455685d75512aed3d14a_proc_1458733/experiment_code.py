import os, pathlib, time, copy, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ---------- mandatory dirs & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH ----------
def locate_spr_bench() -> pathlib.Path:
    guesses = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for g in guesses:
        if g and (g / "train.csv").exists():
            print(f"Found SPR_BENCH at {g.resolve()}")
            return g.resolve()
    raise FileNotFoundError(
        "SPR_BENCH not found; set SPR_DATA_PATH env or place folder here."
    )


# ---------- helpers from baseline ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_struct_complexity(seq: str) -> int:
    return len({tok for tok in seq.strip().split()})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def structure_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_struct_complexity(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------- load dataset ----------
data_path = locate_spr_bench()
spr = load_spr_bench(data_path)

# ---------- vocab & label maps ----------
token_set, label_set = set(), set()
for ex in spr["train"]:
    token_set.update(ex["sequence"].split())
    label_set.add(ex["label"])
token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
label2idx = {lab: i for i, lab in enumerate(sorted(label_set))}
idx2label = {i: l for l, i in label2idx.items()}


# ---------- build graphs ----------
def seq_to_graph(example):
    tokens = example["sequence"].split()
    n = len(tokens)
    x = torch.tensor([token2idx[tok] for tok in tokens], dtype=torch.long).unsqueeze(-1)

    edges, etypes = [], []
    # relation 0: adjacency
    for i in range(n - 1):
        edges.extend([(i, i + 1), (i + 1, i)])
        etypes.extend([0, 0])
    # relation 1: same shape
    shape_map = {}
    for idx, tok in enumerate(tokens):
        shape_map.setdefault(tok[0], []).append(idx)
    for idxs in shape_map.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append((i, j))
                    etypes.append(1)
    # relation 2: same color
    color_map = {}
    for idx, tok in enumerate(tokens):
        if len(tok) > 1:
            color_map.setdefault(tok[1], []).append(idx)
    for idxs in color_map.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append((i, j))
                    etypes.append(2)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(etypes, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    y = torch.tensor([label2idx[example["label"]]], dtype=torch.long)
    data = Data(
        x=x, edge_index=edge_index, edge_type=edge_type, y=y, seq=example["sequence"]
    )
    return data


train_graphs = [seq_to_graph(ex) for ex in spr["train"]]
dev_graphs = [seq_to_graph(ex) for ex in spr["dev"]]
test_graphs = [seq_to_graph(ex) for ex in spr["test"]]

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)


# ---------- model ----------
class SPR_RGCN(nn.Module):
    def __init__(self, vocab, embed_dim, hid, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, embed_dim, padding_idx=0)
        self.conv1 = RGCNConv(embed_dim, hid, num_relations=3)
        self.conv2 = RGCNConv(hid, hid, num_relations=3)
        self.lin = nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.emb(x.squeeze(-1))
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------- training utils ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss, preds, labels, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            tot_loss += loss.item() * batch.num_graphs
            preds.extend(out.argmax(-1).cpu().tolist())
            labels.extend(batch.y.view(-1).cpu().tolist())
            seqs.extend(batch.seq)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    strwa = structure_weighted_accuracy(seqs, labels, preds)
    bwa = (cwa + swa) / 2
    return tot_loss / len(loader.dataset), bwa, cwa, swa, strwa, preds, labels, seqs


# ---------- experiment dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "StrWA": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------- train ----------
model = SPR_RGCN(len(token2idx), 32, 64, len(label2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val, best_state, wait, patience, max_epochs = -1.0, None, 0, 4, 40

for epoch in range(1, max_epochs + 1):
    model.train()
    ep_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * batch.num_graphs
    train_loss = ep_loss / len(train_loader.dataset)
    val_loss, val_bwa, val_cwa, val_swa, val_strwa, _, _, _ = evaluate(
        model, dev_loader
    )
    _, train_bwa, _, _, train_strwa, _, _, _ = evaluate(model, train_loader)

    # logging
    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append(train_bwa)
    ed["metrics"]["val"].append(val_bwa)
    ed["StrWA"]["train"].append(train_strwa)
    ed["StrWA"]["val"].append(val_strwa)
    ed["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"BWA={val_bwa:.4f} CWA={val_cwa:.4f} SWA={val_swa:.4f} StrWA={val_strwa:.4f}"
    )

    # early stop on BWA
    if val_bwa > best_val:
        best_val, best_state, wait = val_bwa, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# restore best
model.load_state_dict(best_state)

# ---------- test ----------
(
    test_loss,
    test_bwa,
    test_cwa,
    test_swa,
    test_strwa,
    test_preds,
    test_labels,
    test_seqs,
) = evaluate(model, test_loader)
ed = experiment_data["SPR_BENCH"]
ed["predictions"] = test_preds
ed["ground_truth"] = test_labels
ed["test_metrics"] = {
    "loss": test_loss,
    "BWA": test_bwa,
    "CWA": test_cwa,
    "SWA": test_swa,
    "StrWA": test_strwa,
}
print(
    f"TEST: loss={test_loss:.4f} BWA={test_bwa:.4f} CWA={test_cwa:.4f} "
    f"SWA={test_swa:.4f} StrWA={test_strwa:.4f}"
)

# ---------- plot ----------
epochs = np.arange(1, len(ed["metrics"]["train"]) + 1)
plt.figure(figsize=(6, 4))
plt.plot(epochs, ed["metrics"]["train"], label="Train BWA")
plt.plot(epochs, ed["metrics"]["val"], label="Dev BWA")
plt.plot(epochs, ed["StrWA"]["train"], "--", label="Train StrWA")
plt.plot(epochs, ed["StrWA"]["val"], "--", label="Dev StrWA")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SPR_RGCN accuracy curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "spr_rgcn_curves.png"))
plt.close()

# ---------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
