import os, pathlib, random, time, numpy as np, torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------------------------------------------------------------------
#  Mandatory working directory & device handling
# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
#  Helpers copied from provided SPR utility (no pandas)
# ---------------------------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------------------------------------------------------------------
#  Load or create dataset
# ---------------------------------------------------------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    data = DatasetDict()
    data["train"] = _load("train.csv")
    data["dev"] = _load("dev.csv")
    data["test"] = _load("test.csv")
    return data


root_path = pathlib.Path(os.getenv("SPR_PATH", "SPR_BENCH"))
if root_path.exists():
    spr = load_spr(root_path)
else:
    # ---------- synthetic tiny data so the code can run anywhere ----------
    shapes = list("ABCD")
    colors = list("rgbc")

    def rand_seq():
        n = random.randint(5, 12)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(n))

    def generate_split(n_rows):
        return {
            "id": list(range(n_rows)),
            "sequence": [rand_seq() for _ in range(n_rows)],
            "label": [random.choice(["0", "1", "2"]) for _ in range(n_rows)],
        }

    spr = DatasetDict(
        {
            "train": Dataset.from_dict(generate_split(500)),
            "dev": Dataset.from_dict(generate_split(100)),
            "test": Dataset.from_dict(generate_split(100)),
        }
    )
print({k: len(v) for k, v in spr.items()})

# ---------------------------------------------------------------------
#  Token & label encoding
# ---------------------------------------------------------------------
shape_set, color_set = set(), set()
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        if len(tok) < 2:
            continue
        shape_set.add(tok[0])
        color_set.add(tok[1])
shape2id = {s: i for i, s in enumerate(sorted(shape_set))}
color2id = {c: i for i, c in enumerate(sorted(color_set))}

label_encoder = LabelEncoder()
label_encoder.fit(spr["train"]["label"])


# ---------------------------------------------------------------------
#  Graph building
# ---------------------------------------------------------------------
def seq_to_graph(seq: str, label: str):
    tokens = seq.split()
    n = len(tokens)
    shape_ids = torch.tensor([shape2id[t[0]] for t in tokens], dtype=torch.long)
    color_ids = torch.tensor([color2id[t[1]] for t in tokens], dtype=torch.long)
    pos = torch.arange(n, dtype=torch.float) / (n - 1 if n > 1 else 1)
    y = torch.tensor([label_encoder.transform([label])[0]], dtype=torch.long)

    # consecutive bidirectional edges
    edge_index = []
    for i in range(n - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(
        shape=shape_ids,
        color=color_ids,
        pos=pos.unsqueeze(-1),
        edge_index=edge_index,
        y=y,
        seq=seq,
    )


def build_graph_dataset(split_dataset):
    return [
        seq_to_graph(seq, lab)
        for seq, lab in zip(split_dataset["sequence"], split_dataset["label"])
    ]


train_graphs = build_graph_dataset(spr["train"])
dev_graphs = build_graph_dataset(spr["dev"])
test_graphs = build_graph_dataset(spr["test"])


# ---------------------------------------------------------------------
#  GNN Model
# ---------------------------------------------------------------------
class SPRGraphNet(torch.nn.Module):
    def __init__(self, n_shapes, n_colors, n_classes, emb_dim=16, hid=32):
        super().__init__()
        self.shape_emb = torch.nn.Embedding(n_shapes, emb_dim)
        self.color_emb = torch.nn.Embedding(n_colors, emb_dim)
        self.lin_pos = torch.nn.Linear(1, emb_dim)
        in_dim = emb_dim * 3
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.classifier = torch.nn.Linear(hid, n_classes)

    def forward(self, data):
        # data.shape,color,pos already on device
        x = torch.cat(
            [
                self.shape_emb(data.shape),
                self.color_emb(data.color),
                self.lin_pos(data.pos),
            ],
            dim=-1,
        )
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)


# ---------------------------------------------------------------------
#  Dataloaders
# ---------------------------------------------------------------------
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)

# ---------------------------------------------------------------------
#  Training setup
# ---------------------------------------------------------------------
model = SPRGraphNet(len(shape2id), len(color2id), len(label_encoder.classes_)).to(
    device
)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# experiment_data skeleton
experiment_data = {
    "spr_bench": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------------------------------------------------------------------
#  Train / Evaluate utilities
# ---------------------------------------------------------------------
@torch.no_grad()
def eval_loader(loader):
    model.eval()
    all_logits, all_labels, all_seqs = [], [], []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        all_logits.append(logits.cpu())
        all_labels.append(data.y.cpu())
        all_seqs.extend(data.seq)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.argmax(torch.cat(all_logits), dim=-1).numpy()
    seqs = all_seqs
    loss = criterion(torch.cat(all_logits), torch.from_numpy(y_true)).item()
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    comp = complexity_weighted_accuracy(seqs, y_true, y_pred)
    return loss, cwa, swa, comp, y_true, y_pred


def train_epoch():
    model.train()
    epoch_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * data.num_graphs
    return epoch_loss / len(train_loader.dataset)


# ---------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------
n_epochs = 10
for epoch in range(1, n_epochs + 1):
    t0 = time.time()
    tr_loss = train_epoch()
    val_loss, _, _, val_compwa, _, _ = eval_loader(dev_loader)

    experiment_data["spr_bench"]["losses"]["train"].append(tr_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)
    experiment_data["spr_bench"]["metrics"]["val"].append(val_compwa)
    experiment_data["spr_bench"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch:02d}: train_loss = {tr_loss:.4f} | "
        f"validation_loss = {val_loss:.4f} | "
        f"Val CompWA = {val_compwa:.4f} | time={time.time()-t0:.1f}s"
    )

# ---------------------------------------------------------------------
#  Final evaluation on test split
# ---------------------------------------------------------------------
test_loss, test_cwa, test_swa, test_compwa, y_true, y_pred = eval_loader(test_loader)
print(
    f"\nTest results -- Loss: {test_loss:.4f} | CWA: {test_cwa:.4f} | "
    f"SWA: {test_swa:.4f} | CompWA: {test_compwa:.4f}"
)

experiment_data["spr_bench"]["predictions"] = y_pred.tolist()
experiment_data["spr_bench"]["ground_truth"] = y_true.tolist()
experiment_data["spr_bench"]["metrics"]["test"] = {
    "loss": test_loss,
    "CWA": test_cwa,
    "SWA": test_swa,
    "CompWA": test_compwa,
}

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Saved experiment data to {os.path.join(working_dir, "experiment_data.npy")}')
