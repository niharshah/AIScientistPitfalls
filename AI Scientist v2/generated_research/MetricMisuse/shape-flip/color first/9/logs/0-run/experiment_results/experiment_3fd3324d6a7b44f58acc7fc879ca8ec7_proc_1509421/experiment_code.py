import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random, pathlib, time, numpy as np, torch, torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#################################################################
# -------- utilities: metrics, data loading, graph build -------#
#################################################################
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def sdwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def load_real_spr(root_path: str):
    csv_path = pathlib.Path(root_path)
    train_csv = csv_path / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError
    import pandas as pd

    d_tr = pd.read_csv(train_csv)
    d_dev = pd.read_csv(csv_path / "dev.csv")
    d_te = pd.read_csv(csv_path / "test.csv")
    return d_tr, d_dev, d_te


def make_synthetic_split(n):
    shapes, colors = list("ABC"), list("012")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(5, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        label = random.randint(0, 3)  # 4 fake classes
        seqs.append(seq)
        labels.append(label)
    return seqs, labels


def build_dataset(seqs, labels, shape2id, color2id):
    data_list = []
    for seq, lbl in zip(seqs, labels):
        tokens = seq.split()
        num_nodes = len(tokens)
        shape_ids = [shape2id[t[0]] for t in tokens]
        color_ids = [color2id[t[1]] for t in tokens]
        pos_feat = [
            i / (num_nodes - 1) if num_nodes > 1 else 0.0 for i in range(num_nodes)
        ]
        x = torch.tensor(list(zip(shape_ids, color_ids, pos_feat)), dtype=torch.float)
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        y = torch.tensor([lbl], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, seq=seq)
        data_list.append(data)
    return data_list


#################################################################
# ----------------------- data acquisition ---------------------#
#################################################################
try:
    real_root = os.environ.get("SPR_PATH", "./SPR_BENCH")
    df_train, df_dev, df_test = load_real_spr(real_root)
    train_seqs, train_labels = df_train.sequence.tolist(), df_train.label.tolist()
    dev_seqs, dev_labels = df_dev.sequence.tolist(), df_dev.label.tolist()
    test_seqs, test_labels = df_test.sequence.tolist(), df_test.label.tolist()
    print("Loaded real SPR_BENCH dataset")
except Exception as e:
    print("Real dataset not found, creating synthetic toy data:", e)
    train_seqs, train_labels = make_synthetic_split(2000)
    dev_seqs, dev_labels = make_synthetic_split(400)
    test_seqs, test_labels = make_synthetic_split(400)

all_tokens = (t for seq in train_seqs for t in seq.split())
all_shapes = sorted({tok[0] for tok in all_tokens})
all_colors = sorted({tok[1] for tok in train_seqs[0].split()})
shape2id = {s: i for i, s in enumerate(all_shapes)}
color2id = {c: i for i, c in enumerate(all_colors)}
num_shapes, num_colors = len(shape2id), len(color2id)
num_classes = len(set(train_labels))

train_data = build_dataset(train_seqs, train_labels, shape2id, color2id)
dev_data = build_dataset(dev_seqs, dev_labels, shape2id, color2id)
test_data = build_dataset(test_seqs, test_labels, shape2id, color2id)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)


#################################################################
# ------------------- model definition -------------------------#
#################################################################
class GNNClassifier(nn.Module):
    def __init__(self, num_shapes, num_colors, num_classes):
        super().__init__()
        self.shape_emb = nn.Embedding(num_shapes, 16)
        self.color_emb = nn.Embedding(num_colors, 16)
        self.conv1 = GCNConv(16 + 16 + 1, 32)
        self.conv2 = GCNConv(32, 32)
        self.lin = nn.Linear(32, num_classes)

    def forward(self, data):
        x = data.x
        shape_id = x[:, 0].long()
        color_id = x[:, 1].long()
        pos = x[:, 2].unsqueeze(1)
        x = torch.cat([self.shape_emb(shape_id), self.color_emb(color_id), pos], dim=1)
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


model = GNNClassifier(num_shapes, num_colors, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#################################################################
# ----------------------- training loop ------------------------#
#################################################################
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_sdwa": [], "val_sdwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


def run_loader(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, y_true, y_pred, seqs = 0.0, [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(batch.y.cpu().numpy())
        seqs.extend(batch.seq)
    avg_loss = total_loss / len(loader.dataset)
    acc_sdwa = sdwa(seqs, y_true, y_pred)
    return avg_loss, acc_sdwa, y_true, y_pred


EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_sdwa, _, _ = run_loader(train_loader, train=True)
    val_loss, val_sdwa, y_t, y_p = run_loader(dev_loader, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_sdwa"].append(tr_sdwa)
    experiment_data["SPR_BENCH"]["metrics"]["val_sdwa"].append(val_sdwa)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_SDWA = {val_sdwa:.4f}")

#################################################################
# ----------------------- final test  --------------------------#
#################################################################
test_loss, test_sdwa, gt, pr = run_loader(test_loader, train=False)
print(f"Test   : loss = {test_loss:.4f} | SDWA = {test_sdwa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = pr
experiment_data["SPR_BENCH"]["ground_truth"] = gt

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
