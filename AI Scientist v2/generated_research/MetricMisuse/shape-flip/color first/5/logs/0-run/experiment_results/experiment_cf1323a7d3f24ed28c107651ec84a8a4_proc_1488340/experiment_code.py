import os, random, string, time, pathlib, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict
import warnings

warnings.filterwarnings("ignore")

# ----------------------------- working dir -----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------- device ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------- experiment log --------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_CpxWA": [], "val_CpxWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ----------------------------- util functions --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def build_synthetic_bench(root: pathlib.Path):
    print("Building synthetic SPR_BENCH for demo …")
    root.mkdir(parents=True, exist_ok=True)
    shapes, colors = list("ABCDEF"), list("xyzuvw")

    def gen_split(n_rows: int, fname: str):
        rows = []
        for i in range(n_rows):
            length = random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            label = str(
                sum(token[0] in "ABC" for token in seq.split()) % 3
            )  # dummy rule
            rows.append(f"{i},{seq},{label}\n")
        (root / fname).write_text("id,sequence,label\n" + "".join(rows))

    gen_split(200, "train.csv")
    gen_split(80, "dev.csv")
    gen_split(80, "test.csv")


def unique_shapes(seq: str):
    return len({tok[0] for tok in seq.strip().split() if tok})


def unique_colors(seq: str):
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [unique_shapes(s) * unique_colors(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def build_vocab(dsets):
    shapes, colors = set(), set()
    for split in dsets.values():
        for s in split["sequence"]:
            for tok in s.split():
                if len(tok) >= 2:
                    shapes.add(tok[0])
                    colors.add(tok[1])
    shape2idx = {s: i + 1 for i, s in enumerate(sorted(shapes))}  # 0 reserved
    color2idx = {c: i + 1 for i, c in enumerate(sorted(colors))}
    return shape2idx, color2idx


def seq_to_graph(seq, label, shape2idx, color2idx):
    tokens = seq.split()
    n = len(tokens)
    shape_ids = [shape2idx.get(tok[0], 0) for tok in tokens]
    color_ids = [color2idx.get(tok[1], 0) if len(tok) >= 2 else 0 for tok in tokens]
    edge_index = []
    for i in range(n - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(
        shape_id=torch.tensor(shape_ids, dtype=torch.long),
        color_id=torch.tensor(color_ids, dtype=torch.long),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        seq=seq,
    )
    return data


# ----------------------------- data ------------------------------------------
root_path = pathlib.Path(os.environ.get("SPR_PATH", "./SPR_BENCH"))
if not (root_path / "train.csv").exists():
    build_synthetic_bench(root_path)
spr = load_spr_bench(root_path)
shape2idx, color2idx = build_vocab(spr)

label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
num_classes = len(label2idx)


def make_dataset(split):
    return [
        seq_to_graph(s, label2idx[l], shape2idx, color2idx)
        for s, l in zip(spr[split]["sequence"], spr[split]["label"])
    ]


train_data = make_dataset("train")
val_data = make_dataset("dev")
test_data = make_dataset("test")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)


# ----------------------------- model -----------------------------------------
class SPRGNN(nn.Module):
    def __init__(self, n_shape, n_color, hidden=64, emb=32, num_classes=3):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb, padding_idx=0)
        self.conv1 = GCNConv(emb, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = self.shape_emb(data.shape_id) + self.color_emb(data.color_id)
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = SPRGNN(len(shape2idx), len(color2idx), num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ----------------------------- training loop ---------------------------------
def run_epoch(loader, train_flag=False):
    if train_flag:
        model.train()
    else:
        model.eval()
    total_loss, ys, preds, seqs = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        if train_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        ys.extend(batch.y.view(-1).cpu().tolist())
        preds.extend(out.argmax(dim=1).cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    cpxwa = complexity_weighted_accuracy(seqs, ys, preds)
    return avg_loss, cpxwa, preds, ys


EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    train_loss, train_cpx, _, _ = run_epoch(train_loader, True)
    val_loss, val_cpx, v_pred, v_true = run_epoch(val_loader, False)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | Val CpxWA={val_cpx:.4f}")

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_CpxWA"].append(train_cpx)
    experiment_data["SPR_BENCH"]["metrics"]["val_CpxWA"].append(val_cpx)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

# ----------------------------- final test ------------------------------------
test_loss, test_cpx, t_pred, t_true = run_epoch(test_loader, False)
print(f"\nTest Complexity-Weighted Accuracy: {test_cpx:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = t_pred
experiment_data["SPR_BENCH"]["ground_truth"] = t_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
