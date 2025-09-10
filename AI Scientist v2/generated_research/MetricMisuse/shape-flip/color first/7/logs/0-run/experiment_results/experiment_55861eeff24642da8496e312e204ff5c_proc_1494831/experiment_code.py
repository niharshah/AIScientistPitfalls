import os, time, math, random, numpy as np, torch
from torch import nn
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

# ---------- basic set-up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def complexity_weight(seq: str) -> int:
    return count_color_variety(seq) + count_shape_variety(seq)


def comp_weighted_acc(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1, sum(w))


# ---------- SPR loader (real path or toy fallback) ----------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path: str) -> DatasetDict:
    if os.path.isdir(path):

        def _ld(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        print("Loaded SPR_BENCH from", path)
        return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))

    # toy synthetic fallback (parity of colour digits)
    print("SPR_BENCH not found – generating toy data")
    shapes = list("ABCDEFGH")
    colors = [str(i) for i in range(6)]

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):
        return sum(int(tok[1:]) for tok in seq.split()) % 2

    def mk_split(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": [rule(s) for s in seqs]}
        )

    return DatasetDict(train=mk_split(2000), dev=mk_split(400), test=mk_split(400))


spr = load_spr(SPR_PATH)

# ---------- vocabularies ----------
shape_vocab, color_vocab = {}, {}
max_len = 0
for seq in spr["train"]["sequence"]:
    toks = seq.split()
    max_len = max(max_len, len(toks))
    for tok in toks:
        if tok[0] not in shape_vocab:
            shape_vocab[tok[0]] = len(shape_vocab)
        col = tok[1:] if len(tok) > 1 else ""
        if col not in color_vocab:
            color_vocab[col] = len(color_vocab)
pos_limit = max_len + 1


# ---------- sequence → graph ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    shape_id = [shape_vocab[t[0]] for t in toks]
    color_id = [color_vocab[t[1:] if len(t) > 1 else ""] for t in toks]
    pos_id = list(range(n))

    # sequential edges
    edges = {(i, i + 1) for i in range(n - 1)} | {(i + 1, i) for i in range(n - 1)}
    # same shape
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i][0] == toks[j][0]:
                edges.add((i, j))
                edges.add((j, i))
    # same colour
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i][1:] == toks[j][1:]:
                edges.add((i, j))
                edges.add((j, i))

    if edges:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(
        shape_id=torch.tensor(shape_id, dtype=torch.long),
        color_id=torch.tensor(color_id, dtype=torch.long),
        pos_id=torch.tensor(pos_id, dtype=torch.long),
        edge_index=edge_index,
        y=torch.tensor(label, dtype=torch.long),
        seq=seq,
    )


def encode_split(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode_split, (spr["train"], spr["dev"], spr["test"])
)

train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)


# ---------- model ----------
class GNNClassifier(nn.Module):
    def __init__(self, n_shape, n_color, pos_max, hid=64, out_dim=2):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, 16)
        self.color_emb = nn.Embedding(n_color, 16)
        self.pos_emb = nn.Embedding(pos_max, 8)
        in_dim = 16 + 16 + 8  # 40
        self.conv1 = SAGEConv(in_dim, hid)
        self.bn1 = BatchNorm(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2 = BatchNorm(hid)
        self.lin = nn.Linear(hid, out_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, data):
        x = torch.cat(
            [
                self.shape_emb(data.shape_id),
                self.color_emb(data.color_id),
                self.pos_emb(data.pos_id),
            ],
            dim=-1,
        )
        x = torch.relu(self.bn1(self.conv1(x, data.edge_index)))
        x = self.drop(x)
        x = torch.relu(self.bn2(self.conv2(x, data.edge_index)))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


num_classes = len(set(spr["train"]["label"]))
model = GNNClassifier(
    len(shape_vocab), len(color_vocab), pos_limit, hid=64, out_dim=num_classes
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

# ---------- experiment log ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
        "best_epoch": None,
    }
}


# ---------- training / evaluation helpers ----------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss = tot_correct = tot = 0
    seqs_all, pred_all, true_all = [], [], []
    for bt in loader:
        bt = bt.to(device)
        out = model(bt)
        loss = criterion(out, bt.y)
        tot_loss += loss.item() * bt.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        gts = bt.y.cpu().tolist()
        tot_correct += sum(p == g for p, g in zip(preds, gts))
        tot += bt.num_graphs
        seqs_all.extend(bt.seq)
        pred_all.extend(preds)
        true_all.extend(gts)
    compwa = comp_weighted_acc(seqs_all, true_all, pred_all)
    return tot_loss / tot, tot_correct / tot, compwa, pred_all, true_all, seqs_all


def train_epoch(loader):
    model.train()
    tot_loss = tot_correct = tot = 0
    for bt in loader:
        bt = bt.to(device)
        optimizer.zero_grad()
        out = model(bt)
        loss = criterion(out, bt.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * bt.num_graphs
        preds = out.argmax(dim=-1).cpu().tolist()
        gts = bt.y.cpu().tolist()
        tot_correct += sum(p == g for p, g in zip(preds, gts))
        tot += bt.num_graphs
    return tot_loss / tot, tot_correct / tot


# ---------- training loop ----------
max_epochs, patience = 60, 10
best_val, best_state, pat = math.inf, None, 0
start_time = time.time()
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_acc = train_epoch(train_loader)
    val_loss, val_acc, val_cwa, _, _, _ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"acc": tr_acc})
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "CompWA": val_cwa}
    )

    print(
        f"Epoch {epoch:02d}: validation_loss = {val_loss:.4f} | "
        f"val_acc = {val_acc:.3f} | CompWA = {val_cwa:.3f}"
    )

    if val_loss < best_val - 1e-4:
        best_val, best_state = val_loss, model.state_dict()
        experiment_data["SPR_BENCH"]["best_epoch"] = epoch
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping.")
            break

# ---------- testing ----------
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc, test_cwa, preds, gts, seqs = evaluate(test_loader)
print(f"TEST -- loss:{test_loss:.4f}  acc:{test_acc:.3f}  CompWA:{test_cwa:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["sequences"] = seqs

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
print(f"Total run-time: {time.time()-start_time:.1f}s")
