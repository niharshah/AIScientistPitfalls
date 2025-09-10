import os, pathlib, random, time, json, math, itertools, collections
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# -------------------------------------------------------------
# housekeeping & device
# -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------
# SPR helpers (taken from the given utility)
# -------------------------------------------------------------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [cw if t == p else 0 for cw, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [sw if t == p else 0 for sw, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) or 1)


def complexity_adjusted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    corr = [cw if t == p else 0 for cw, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) or 1)


# -------------------------------------------------------------
# Fallback synthetic data (runs quickly if real data missing)
# -------------------------------------------------------------
def generate_synthetic_csv(path, n):
    shapes = list("ABCDEFG")
    colors = list("123456")
    labels = ["X", "Y", "Z"]
    with open(path, "w") as f:
        f.write("id,sequence,label\n")
        for i in range(n):
            seq_len = random.randint(4, 10)
            tokens = [
                "{}{}".format(random.choice(shapes), random.choice(colors))
                for _ in range(seq_len)
            ]
            label = random.choice(labels)
            f.write(f"{i},{' '.join(tokens)},{label}\n")


def ensure_dataset():
    root = pathlib.Path("./SPR_BENCH")
    if root.exists():
        return root
    print("SPR_BENCH not found, creating tiny synthetic data …")
    root.mkdir(exist_ok=True)
    generate_synthetic_csv(root / "train.csv", 2000)
    generate_synthetic_csv(root / "dev.csv", 400)
    generate_synthetic_csv(root / "test.csv", 400)
    return root


DATA_PATH = ensure_dataset()
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# -------------------------------------------------------------
# Vocabulary & label mapping
# -------------------------------------------------------------
def build_vocab(dataset):
    vocab = set()
    for seq in dataset["sequence"]:
        vocab.update(seq.strip().split())
    stoi = {tok: i + 1 for i, tok in enumerate(sorted(vocab))}  # 0 = PAD
    return stoi


token2idx = build_vocab(dsets["train"])
num_tokens = len(token2idx) + 1  # including PAD


def build_labelmap(dataset):
    labels = sorted(set(dataset["label"]))
    return {lbl: i for i, lbl in enumerate(labels)}


label2idx = build_labelmap(dsets["train"])
num_classes = len(label2idx)


# -------------------------------------------------------------
# PyG Dataset
# -------------------------------------------------------------
def seq_to_graph(sequence, y):
    toks = sequence.strip().split()
    idxs = [token2idx[t] for t in toks]
    x = torch.tensor(idxs, dtype=torch.long).unsqueeze(-1)  # (N,1)
    if len(toks) == 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src = torch.arange(0, len(toks) - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label2idx[y]], dtype=torch.long),
        seq_raw=sequence,
    )
    return data


class SPRGraphDataset(InMemoryDataset):
    def __init__(self, hf_split):
        self.hf_split = hf_split
        data_list = [
            seq_to_graph(s, l) for s, l in zip(hf_split["sequence"], hf_split["label"])
        ]
        super().__init__(".", None, None, None)
        self.data, self.slices = self.collate(data_list)


train_dataset = SPRGraphDataset(dsets["train"])
dev_dataset = SPRGraphDataset(dsets["dev"])
test_dataset = SPRGraphDataset(dsets["test"])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# -------------------------------------------------------------
# Model
# -------------------------------------------------------------
class GCNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())  # (N, d)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        out = global_mean_pool(x, data.batch)  # (batch, d)
        return self.lin(out)


model = GCNClassifier(num_tokens, 64, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------------
# experiment_data dict
# -------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# -------------------------------------------------------------
# Training / Evaluation
# -------------------------------------------------------------
def run_epoch(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_seq, all_true, all_pred = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1).detach().cpu().numpy()
        ys = batch.y.view(-1).cpu().numpy()
        total_correct += (preds == ys).sum()
        total_samples += batch.num_graphs
        all_true.extend(ys)
        all_pred.extend(preds)
        all_seq.extend(batch.seq_raw)
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    caa = complexity_adjusted_accuracy(all_seq, all_true, all_pred)
    return avg_loss, acc, cwa, swa, caa, all_seq, all_true, all_pred


num_epochs = 8
for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_cwa, tr_swa, tr_caa, *_ = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_cwa, val_swa, val_caa, *_ = run_epoch(
        dev_loader, train=False
    )

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"Val-Acc {val_acc:.3f}  CAA {val_caa:.3f}"
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"acc": tr_acc, "cwa": tr_cwa, "swa": tr_swa, "caa": tr_caa}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "cwa": val_cwa, "swa": val_swa, "caa": val_caa}
    )
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

# Final test evaluation
test_loss, test_acc, test_cwa, test_swa, test_caa, seqs, y_true, y_pred = run_epoch(
    test_loader, train=False
)
print(
    f"\nTEST  | loss {test_loss:.4f}  acc {test_acc:.3f}  "
    f"CWA {test_cwa:.3f}  SWA {test_swa:.3f}  CAA {test_caa:.3f}"
)

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true

# -------------------------------------------------------------
# Save results
# -------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
