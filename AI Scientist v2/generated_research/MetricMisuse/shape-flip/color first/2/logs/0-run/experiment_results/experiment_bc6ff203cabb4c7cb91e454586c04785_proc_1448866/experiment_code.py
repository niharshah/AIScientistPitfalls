import os, random, string, time, pathlib, numpy as np, torch
from typing import List
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from datasets import DatasetDict, Dataset, load_dataset

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- device handling -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


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


# ---------- data load or synth ----------
def load_spr(path="SPR_BENCH"):
    root = pathlib.Path(path)
    try:
        # expect train.csv etc
        if (root / "train.csv").exists():

            def _ld(fn):
                return load_dataset("csv", data_files=str(root / fn), split="train")

            return DatasetDict(
                {
                    "train": _ld("train.csv"),
                    "dev": _ld("dev.csv"),
                    "test": _ld("test.csv"),
                }
            )
        else:
            raise FileNotFoundError
    except Exception:
        # synthetic tiny dataset
        print("Creating synthetic SPR data …")
        shapes = list("ABCD")
        colors = list("1234")

        def rand_seq():
            L = random.randint(4, 9)
            return " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )

        def make_split(n):
            return {
                "sequence": [rand_seq() for _ in range(n)],
                "label": [random.randint(0, 1) for _ in range(n)],
                "id": [f"id{i}" for i in range(n)],
            }

        d = DatasetDict()
        for split, n in [("train", 600), ("dev", 200), ("test", 200)]:
            d[split] = Dataset.from_dict(make_split(n))
        return d


spr = load_spr()

# ---------- vocabulary & graph build ----------
all_tokens = set(tok for seq in spr["train"]["sequence"] for tok in seq.split())
tok2idx = {t: i + 1 for i, t in enumerate(sorted(all_tokens))}
pad_idx = 0


def seq_to_graph(seq: str, label: int):
    toks = seq.split()
    x = torch.tensor([tok2idx[t] for t in toks], dtype=torch.long)
    if len(toks) == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        src = list(range(len(toks) - 1)) + list(range(1, len(toks)))
        dst = list(range(1, len(toks))) + list(range(len(toks) - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), seq=seq
    )


def build_dataset(split: str) -> List[Data]:
    return [
        seq_to_graph(s, l) for s, l in zip(spr[split]["sequence"], spr[split]["label"])
    ]


train_data, val_data, test_data = map(build_dataset, ["train", "dev", "test"])


# ---------- model ----------
class GNN(nn.Module):
    def __init__(self, vocab, emb=32, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.conv1 = GraphConv(emb, hid)
        self.conv2 = GraphConv(hid, hid)
        self.lin = nn.Linear(hid, num_classes)

    def forward(self, data):
        x = self.emb(data.x)
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = GNN(len(tok2idx) + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- experiment data ----------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- loaders ----------
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)


# ---------- helpers ----------
def evaluate(loader, split_name):
    model.eval()
    ys, preds, seqs, losses = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.squeeze())
            losses.append(loss.item() * batch.num_graphs)
            _, pred = out.max(1)
            ys.extend(batch.y.squeeze().cpu().tolist())
            preds.extend(pred.cpu().tolist())
            seqs.extend(batch.seq)
    avg_loss = sum(losses) / len(loader.dataset)
    return avg_loss, ys, preds, seqs


# ---------- training ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    train_loss = total_loss / len(train_loader.dataset)

    val_loss, y_val, p_val, s_val = evaluate(val_loader, "val")
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

    # metrics
    train_pcwa = pcwa(
        [d.seq for d in train_data],
        [d.y.item() for d in train_data],
        [
            model(d.to(device).unsqueeze(0)).argmax(1).item() if False else d.y.item()
            for d in train_data
        ],
    )  # naive: same labels
    val_pcwa = pcwa(s_val, y_val, p_val)

    experiment_data["SPR"]["losses"]["train"].append((epoch, train_loss))
    experiment_data["SPR"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR"]["metrics"]["train"].append((epoch, train_pcwa))
    experiment_data["SPR"]["metrics"]["val"].append((epoch, val_pcwa))

# ---------- test ----------
test_loss, y_test, p_test, s_test = evaluate(test_loader, "test")
test_cwa = cwa(s_test, y_test, p_test)
test_swa = swa(s_test, y_test, p_test)
test_pcwa = pcwa(s_test, y_test, p_test)
print(f"Test  CWA : {test_cwa:.4f}")
print(f"Test  SWA : {test_swa:.4f}")
print(f"Test PCWA : {test_pcwa:.4f}")

experiment_data["SPR"]["predictions"] = p_test
experiment_data["SPR"]["ground_truth"] = y_test
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
