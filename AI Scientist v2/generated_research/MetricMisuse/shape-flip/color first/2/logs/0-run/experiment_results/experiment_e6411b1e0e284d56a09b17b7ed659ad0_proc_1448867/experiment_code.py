import os, random, pathlib, numpy as np, torch, time
from typing import List
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from datasets import load_dataset, Dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        1, sum(w)
    )


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        1, sum(w)
    )


def pcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        1, sum(w)
    )


# ---------- data ----------
def load_spr(root="SPR_BENCH") -> DatasetDict:
    root = pathlib.Path(root)
    if (root / "train.csv").exists():

        def _ld(fn):
            return load_dataset("csv", data_files=str(root / fn), split="train")

        return DatasetDict(
            {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
        )
    # ---- fallback synthetic tiny dataset ----
    print("Dataset not found, generating synthetic data.")
    shapes, colors = list("ABCD"), list("1234")

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def make_split(n):
        return {
            "id": [f"id{i}" for i in range(n)],
            "sequence": [rand_seq() for _ in range(n)],
            "label": [random.randint(0, 1) for _ in range(n)],
        }

    d = DatasetDict()
    for sp, n in [("train", 800), ("dev", 200), ("test", 200)]:
        d[sp] = Dataset.from_dict(make_split(n))
    return d


spr = load_spr()

# ---------- vocab ----------
all_tokens = set(tok for seq in spr["train"]["sequence"] for tok in seq.split())
tok2idx = {t: i + 1 for i, t in enumerate(sorted(all_tokens))}
pad_idx = 0


# ---------- graph construction ----------
def seq_to_graph(seq: str, label: int) -> Data:
    toks = seq.split()
    x = torch.tensor([tok2idx[t] for t in toks], dtype=torch.long)
    edge_src, edge_dst = [], []
    # sequential edges
    for i in range(len(toks) - 1):
        edge_src.extend([i, i + 1])
        edge_dst.extend([i + 1, i])
    # same colour / same shape edges
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            if toks[i][0] == toks[j][0] or toks[i][1] == toks[j][1]:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
    if not edge_src:
        edge_src, edge_dst = [0], [0]
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), seq=seq
    )


def build_dataset(split: str) -> List[Data]:
    return [
        seq_to_graph(s, l) for s, l in zip(spr[split]["sequence"], spr[split]["label"])
    ]


train_data, val_data, test_data = map(build_dataset, ["train", "dev", "test"])


# ---------- model ----------
class SPRGNN(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, num_classes=2, heads=4, drop=0.25):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.conv1 = GATConv(emb, hid // heads, heads=heads, dropout=drop)
        self.conv2 = GATConv(hid, hid // heads, heads=heads, dropout=drop)
        self.conv3 = GATConv(hid, hid // heads, heads=heads, dropout=drop)
        self.lin = nn.Sequential(nn.Dropout(drop), nn.Linear(hid, num_classes))

    def forward(self, data):
        x = self.emb(data.x)
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = torch.relu(self.conv3(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = SPRGNN(len(tok2idx) + 1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
criterion = nn.CrossEntropyLoss()
# ---------- loaders ----------
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# ---------- experiment store ----------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- evaluation ----------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    losses = []
    ys = []
    preds = []
    seqs = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y.squeeze())
        losses.append(loss.item() * batch.num_graphs)
        pred = out.argmax(1)
        ys.extend(batch.y.squeeze().cpu().tolist())
        preds.extend(pred.cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = sum(losses) / len(loader.dataset)
    return avg_loss, ys, preds, seqs


# ---------- training ----------
best_val_loss = float("inf")
patience, pat_count = 4, 0
epochs = 20
sample_train_idx = random.sample(range(len(train_data)), min(800, len(train_data)))
sample_train = [train_data[i] for i in sample_train_idx]

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    train_loss = epoch_loss / len(train_loader.dataset)
    # metrics on sampled train subset
    tr_loss_tmp, y_tr, p_tr, s_tr = evaluate(DataLoader(sample_train, batch_size=256))
    tr_pcwa = pcwa(s_tr, y_tr, p_tr)
    # validation
    val_loss, y_val, p_val, s_val = evaluate(val_loader)
    val_pcwa = pcwa(s_val, y_val, p_val)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_PCWA={val_pcwa:.4f}")
    experiment_data["SPR"]["losses"]["train"].append((epoch, train_loss))
    experiment_data["SPR"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR"]["metrics"]["train"].append((epoch, tr_pcwa))
    experiment_data["SPR"]["metrics"]["val"].append((epoch, val_pcwa))

    # early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        pat_count = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        pat_count += 1
    if pat_count >= patience:
        print("Early stopping triggered.")
        break
    scheduler.step()

# ---------- test ----------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
test_loss, y_test, p_test, s_test = evaluate(test_loader)
test_cwa, test_swa, test_pcwa = (
    cwa(s_test, y_test, p_test),
    swa(s_test, y_test, p_test),
    pcwa(s_test, y_test, p_test),
)
print(f"Test  CWA : {test_cwa:.4f}")
print(f"Test  SWA : {test_swa:.4f}")
print(f"Test PCWA : {test_pcwa:.4f}")

experiment_data["SPR"]["predictions"] = p_test
experiment_data["SPR"]["ground_truth"] = y_test
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
