import os, time, copy, pathlib, numpy as np, torch, torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------- #
# Required boiler-plate
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------- #
# Dataset helpers (taken from baseline)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def extract_tokens(seq: str):
    return seq.strip().split()


def color_of(tok):
    return tok[1] if len(tok) > 1 else "?"


def shape_of(tok):
    return tok[0] if tok else "?"


def count_color_variety(seq):
    return len({color_of(t) for t in extract_tokens(seq)})


def count_shape_variety(seq):
    return len({shape_of(t) for t in extract_tokens(seq)})


def count_struct_complex(seq):
    return len({(shape_of(t), color_of(t)) for t in extract_tokens(seq)})


def CWA(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_t, y_p) if yt == yp) / max(sum(w), 1)


def SWA(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_t, y_p) if yt == yp) / max(sum(w), 1)


def StrWA(seqs, y_t, y_p):
    w = [count_struct_complex(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_t, y_p) if yt == yp) / max(sum(w), 1)


# -------------------------------------------------------------------- #
def locate_data():
    cands = [
        pathlib.Path(p)
        for p in [
            "./SPR_BENCH",
            "../SPR_BENCH",
            "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
            os.getenv("SPR_DATA_PATH", ""),
        ]
    ]
    for p in cands:
        if p and (p / "train.csv").exists():
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH not found")


data_root = locate_data()
dsets = load_spr_bench(data_root)
print("Loaded dataset.")

# -------------------------------------------------------------------- #
# Vocabulary + label mapping
token_set, set_labels = set(), set()
for ex in dsets["train"]:
    token_set.update(extract_tokens(ex["sequence"]))
    set_labels.add(ex["label"])
token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
label2idx = {lab: i for i, lab in enumerate(sorted(set_labels))}
idx2label = {i: l for l, i in label2idx.items()}


def seq_to_graph(example):
    tokens = extract_tokens(example["sequence"])
    n = len(tokens)
    x = torch.tensor([token2idx[t] for t in tokens], dtype=torch.long).unsqueeze(-1)
    edges = []
    # sequential
    edges.extend([(i, i + 1) for i in range(n - 1)])
    # same color / same shape
    color_groups = {}, {}
    from collections import defaultdict

    color_groups = defaultdict(list)
    shape_groups = defaultdict(list)
    for i, t in enumerate(tokens):
        color_groups[color_of(t)].append(i)
        shape_groups[shape_of(t)].append(i)
    for group in list(color_groups.values()) + list(shape_groups.values()):
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    edges.append((group[i], group[j]))
    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor(
            [list(src) + list(dst), list(dst) + list(src)], dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[example["label"]]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, seq=example["sequence"])
    return data


train_graphs = [seq_to_graph(ex) for ex in dsets["train"]]
dev_graphs = [seq_to_graph(ex) for ex in dsets["dev"]]
test_graphs = [seq_to_graph(ex) for ex in dsets["test"]]

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=128, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)


# -------------------------------------------------------------------- #
class SPR_GCN(nn.Module):
    def __init__(self, vocab, embed=32, num_cls=10):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, embed, padding_idx=0)
        self.conv1 = GCNConv(embed, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin = nn.Linear(64, num_cls)

    def forward(self, x, edge_index, batch):
        x = self.emb(x.squeeze(-1))
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = SPR_GCN(len(token2idx), embed=32, num_cls=len(label2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "StrWA": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}
max_epochs, patience = 30, 5
best_val, best_state = -1, None
wait = 0


def run_eval(loader):
    model.eval()
    tot_loss, all_pred, all_y, seqs = 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            tot_loss += loss.item() * batch.num_graphs
            preds = out.argmax(-1).cpu().tolist()
            labs = batch.y.view(-1).cpu().tolist()
            all_pred.extend(preds)
            all_y.extend(labs)
            seqs.extend(batch.seq)
    cwa = CWA(seqs, all_y, all_pred)
    swa = SWA(seqs, all_y, all_pred)
    strwa = StrWA(seqs, all_y, all_pred)
    bwa = (cwa + swa) / 2
    return tot_loss / len(loader.dataset), bwa, cwa, swa, strwa, all_pred, all_y


for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    train_loss = epoch_loss / len(train_loader.dataset)
    val_loss, val_bwa, val_cwa, val_swa, val_strwa, _, _ = run_eval(dev_loader)
    train_loss_eval, train_bwa, _, _, train_strwa, _, _ = run_eval(train_loader)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | BWA={val_bwa:.3f} CWA={val_cwa:.3f} SWA={val_swa:.3f} StrWA={val_strwa:.3f}"
    )
    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss_eval)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append(train_bwa)
    ed["metrics"]["val"].append(val_bwa)
    ed["StrWA"]["train"].append(train_strwa)
    ed["StrWA"]["val"].append(val_strwa)

    if val_bwa > best_val:
        best_val, val_state = val_bwa, copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# restore best
model.load_state_dict(val_state)
test_loss, test_bwa, test_cwa, test_swa, test_strwa, test_pred, test_y = run_eval(
    test_loader
)
print(
    f"Test -> BWA:{test_bwa:.3f} CWA:{test_cwa:.3f} SWA:{test_swa:.3f} StrWA:{test_strwa:.3f}"
)

experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_y
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
