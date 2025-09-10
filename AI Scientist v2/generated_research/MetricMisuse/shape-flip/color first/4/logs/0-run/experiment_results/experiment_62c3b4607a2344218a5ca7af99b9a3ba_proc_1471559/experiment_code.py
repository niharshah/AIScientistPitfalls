import os, random, string, time, math, itertools, json, pathlib, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- metrics store ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ---------- helper: complexity weight ----------
def complexity_weight(seq: str):
    shapes = {tok[0] for tok in seq.split() if tok}
    colors = {tok[1] for tok in seq.split() if len(tok) > 1}
    return len(shapes) * len(colors)


def cwa2(seqs, y_true, y_pred):
    weights = [complexity_weight(s) for s in seqs]
    num = sum(w for w, t, p in zip(weights, y_true, y_pred) if t == p)
    return num / (sum(weights) + 1e-9)


# ---------- load or synthesize dataset ----------
def load_real_dataset():
    from datasets import load_dataset

    root = os.getenv("SPR_DATA_PATH", "./SPR_BENCH")
    root_path = pathlib.Path(root)
    if not (root_path / "train.csv").exists():
        raise FileNotFoundError

    def _load(csv):
        return load_dataset("csv", data_files=str(root_path / csv), split="train")

    return {_split: _load(f"{_split}.csv") for _split in ["train", "dev", "test"]}


def make_synthetic(n):
    shapes = list("RST")  # 3 shapes
    colors = list("ABC")  # 3 colors
    seqs, labels = [], []
    for i in range(n):
        length = random.randint(4, 8)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(length)]
        seq = " ".join(toks)
        # simple rule: label is parity of distinct colors
        lbl = len({t[1] for t in toks}) % 2
        seqs.append(seq)
        labels.append(lbl)
    return {"sequence": seqs, "label": labels}


def load_dataset_all():
    try:
        data = load_real_dataset()
        print("Loaded real SPR_BENCH dataset.")
        return data
    except Exception as e:
        print("Real dataset not found, generating synthetic toy data.")
        return {
            "train": make_synthetic(1000),
            "dev": make_synthetic(200),
            "test": make_synthetic(200),
        }


raw_data = load_dataset_all()

# ---------- vocabulary ----------
all_seqs = list(raw_data["train"]["sequence"]) + list(raw_data["dev"]["sequence"])
shapes = sorted({tok[0] for seq in all_seqs for tok in seq.split()})
colors = sorted({tok[1] for seq in all_seqs for tok in seq.split()})
shape2id = {s: i for i, s in enumerate(shapes)}
color2id = {c: i for i, c in enumerate(colors)}
num_shapes, num_colors = len(shapes), len(colors)
num_classes = len(set(raw_data["train"]["label"]))


# ---------- graph conversion ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    x = torch.tensor([[shape2id[t[0]], color2id[t[1]]] for t in toks], dtype=torch.long)
    edge_index = (
        torch.tensor(
            list(zip(range(n - 1), range(1, n))) + list(zip(range(1, n), range(n - 1))),
            dtype=torch.long,
        )
        .t()
        .contiguous()
        if n > 1
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), seq=seq
    )


def build_dataset(split):
    seqs = raw_data[split]["sequence"]
    labels = raw_data[split]["label"]
    return [seq_to_graph(s, l) for s, l in zip(seqs, labels)]


train_graphs = build_dataset("train")
dev_graphs = build_dataset("dev")


# ---------- DataLoaders ----------
def pyg_collate(data_list):
    return DataLoader.collate(data_list)


train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)


# ---------- model ----------
class SPR_GCN(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 16
        self.shape_emb = nn.Embedding(num_shapes, emb_dim)
        self.color_emb = nn.Embedding(num_colors, emb_dim)
        self.gcn1 = GCNConv(emb_dim * 2, 32)
        self.gcn2 = GCNConv(32, 32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, data):
        x = torch.cat(
            [self.shape_emb(data.x[:, 0]), self.color_emb(data.x[:, 1])], dim=1
        )
        x = F.relu(self.gcn1(x, data.edge_index))
        x = F.relu(self.gcn2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)


model = SPR_GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ---------- train / eval ----------
def run_epoch(loader, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss, y_true, y_pred, seqs = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y.squeeze())
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch.y.squeeze().cpu().tolist())
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    metric = cwa2(seqs, y_true, y_pred)
    return avg_loss, metric, y_true, y_pred


# ---------- training loop ----------
epochs = 10
for epoch in range(1, epochs + 1):
    tr_loss, tr_metric, _, _ = run_epoch(train_loader, True)
    val_loss, val_metric, val_y, val_p = run_epoch(dev_loader, False)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_metric)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_metric)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA2 = {val_metric:.4f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
