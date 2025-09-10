import os, pathlib, time, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import List
from datasets import DatasetDict, load_dataset

# ---------------------------------------------------------------------
# mandatory working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# device ---------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# dataset locating helpers --------------------------------------------
def locate_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in candidates:
        if p and (p / "train.csv").exists() and (p / "dev.csv").exists():
            print(f"Found SPR_BENCH at: {p.resolve()}")
            return p.resolve()
    raise FileNotFoundError(
        "SPR_BENCH dataset not found. Place it in cwd/.. or set SPR_DATA_PATH."
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# ---------------------------------------------------------------------
# metric helpers -------------------------------------------------------
def extract_tokens(seq: str) -> List[str]:
    return seq.strip().split()


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) else 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) else 1)


# ---------------------------------------------------------------------
# load dataset ---------------------------------------------------------
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)

# ---------------------------------------------------------------------
# vocabularies ---------------------------------------------------------
token_set, label_set = set(), set()
for ex in spr["train"]:
    token_set.update(extract_tokens(ex["sequence"]))
    label_set.add(ex["label"])
token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
label2idx = {lab: i for i, lab in enumerate(sorted(label_set))}
idx2label = {i: lab for lab, i in label2idx.items()}


# ---------------------------------------------------------------------
# graph construction ---------------------------------------------------
def seq_to_data(example):
    seq = example["sequence"]
    tokens = extract_tokens(seq)
    node_idx = [token2idx.get(tok, 0) for tok in tokens]
    x = torch.tensor(node_idx, dtype=torch.long).unsqueeze(-1)
    if len(tokens) > 1:
        src = torch.arange(0, len(tokens) - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[example["label"]]], dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, y=y)
    d.seq = seq
    return d


train_graphs = [seq_to_data(ex) for ex in spr["train"]]
dev_graphs = [seq_to_data(ex) for ex in spr["dev"]]
test_graphs = [seq_to_data(ex) for ex in spr["test"]]

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------
# model ----------------------------------------------------------------
class SPR_GCN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.gcn1 = GCNConv(embed_dim, 64)
        self.gcn2 = GCNConv(64, 64)
        self.lin = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.embed(x.squeeze(-1))
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.drop(x)
        return self.lin(x)


criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------
# experiment data dict -------------------------------------------------
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}


# ---------------------------------------------------------------------
# evaluation -----------------------------------------------------------
def evaluate(model: nn.Module, loader):
    model.eval()
    preds, labels, seqs = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for bt in loader:
            bt = bt.to(device)
            out = model(bt.x, bt.edge_index, bt.batch)
            loss = criterion(out, bt.y.view(-1))
            loss_sum += loss.item() * bt.num_graphs
            preds.extend(out.argmax(-1).cpu().tolist())
            labels.extend(bt.y.view(-1).cpu().tolist())
            seqs.extend(bt.seq)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    return loss_sum / len(loader.dataset), (cwa + swa) / 2.0, cwa, swa, preds, labels


# ---------------------------------------------------------------------
# training loop per LR -------------------------------------------------
def train_with_lr(lr: float, num_epochs: int = 5):
    model = SPR_GCN(len(token2idx), 32, len(label2idx)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lr_key = f"{lr:.0e}" if lr < 1 else str(lr)
    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, num_epochs + 1):
        model.train()
        ep_loss = 0.0
        for bt in train_loader:
            bt = bt.to(device)
            optim.zero_grad()
            out = model(bt.x, bt.edge_index, bt.batch)
            loss = criterion(out, bt.y.view(-1))
            loss.backward()
            optim.step()
            ep_loss += loss.item() * bt.num_graphs
        train_loss = ep_loss / len(train_loader.dataset)
        val_loss, val_bwa, val_cwa, val_swa, _, _ = evaluate(model, dev_loader)
        _, train_bwa, _, _, _, _ = evaluate(model, train_loader)

        exp_entry["losses"]["train"].append(train_loss)
        exp_entry["losses"]["val"].append(val_loss)
        exp_entry["metrics"]["train"].append(train_bwa)
        exp_entry["metrics"]["val"].append(val_bwa)
        exp_entry["timestamps"].append(time.time())

        print(
            f"[lr={lr}] Epoch {ep}/{num_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"BWA={val_bwa:.4f} (CWA={val_cwa:.4f}, SWA={val_swa:.4f})"
        )
    experiment_data["learning_rate"]["SPR_BENCH"][lr_key] = exp_entry
    return model, exp_entry["metrics"]["val"][-1]  # final val BWA


# ---------------------------------------------------------------------
# hyperparameter sweep -------------------------------------------------
learning_rates = [3e-4, 5e-4, 1e-3, 3e-3]
best_lr, best_val_bwa, best_model = None, -1.0, None

for lr in learning_rates:
    model, final_val_bwa = train_with_lr(lr, num_epochs=5)
    if final_val_bwa > best_val_bwa:
        best_val_bwa = final_val_bwa
        best_lr = lr
        best_model = model  # keep reference to best model
print(f"\nBest learning rate based on dev BWA: {best_lr} (BWA={best_val_bwa:.4f})")

# ---------------------------------------------------------------------
# test evaluation with best model -------------------------------------
test_loss, test_bwa, test_cwa, test_swa, test_preds, test_labels = evaluate(
    best_model, test_loader
)
print(
    f"Test results with best lr={best_lr}: "
    f"loss={test_loss:.4f} BWA={test_bwa:.4f} "
    f"CWA={test_cwa:.4f} SWA={test_swa:.4f}"
)
best_key = f"{best_lr:.0e}" if best_lr < 1 else str(best_lr)
experiment_data["learning_rate"]["SPR_BENCH"][best_key]["predictions"] = test_preds
experiment_data["learning_rate"]["SPR_BENCH"][best_key]["ground_truth"] = test_labels

# ---------------------------------------------------------------------
# save experiment data -------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ---------------------------------------------------------------------
# plot BWA curve for best lr ------------------------------------------
epochs = np.arange(
    1,
    len(experiment_data["learning_rate"]["SPR_BENCH"][best_key]["metrics"]["val"]) + 1,
)
plt.figure()
plt.plot(
    epochs,
    experiment_data["learning_rate"]["SPR_BENCH"][best_key]["metrics"]["train"],
    label="Train BWA",
)
plt.plot(
    epochs,
    experiment_data["learning_rate"]["SPR_BENCH"][best_key]["metrics"]["val"],
    label="Dev BWA",
)
plt.xlabel("Epoch")
plt.ylabel("BWA")
plt.title(f"BWA over epochs (best lr={best_lr})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "bwa_curve_spr.png"))
print(f"Curve saved to {os.path.join(working_dir, 'bwa_curve_spr.png')}")
