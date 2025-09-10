import os, pathlib, time, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import DatasetDict, load_dataset
from typing import List

# ---------------------------------------------------------------------
# working dir / device ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------
# find SPR-BENCH -------------------------------------------------------
def locate_spr_bench() -> pathlib.Path:
    cands = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in cands:
        if p and (p / "train.csv").exists():
            print("Found SPR_BENCH at:", p.resolve())
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH dataset folder not found.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(split):
        return load_dataset(
            "csv", data_files=str(root / split), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


# ---------------------------------------------------------------------
# metric helpers ------------------------------------------------------
def extract_tokens(seq: str) -> List[str]:
    return seq.strip().split()


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def color_weighted_accuracy(seqs, y, yhat):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yi == pi else 0 for wi, yi, pi in zip(w, y, yhat)) / (sum(w) or 1)


def shape_weighted_accuracy(seqs, y, yhat):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yi == pi else 0 for wi, yi, pi in zip(w, y, yhat)) / (sum(w) or 1)


# ---------------------------------------------------------------------
# data & vocab --------------------------------------------------------
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)

token2idx, label2idx = {}, {}
for ex in spr["train"]:
    for tok in extract_tokens(ex["sequence"]):
        if tok not in token2idx:
            token2idx[tok] = len(token2idx) + 1
    if ex["label"] not in label2idx:
        label2idx[ex["label"]] = len(label2idx)
idx2label = {i: l for l, i in label2idx.items()}


def seq_to_data(example):
    tokens = extract_tokens(example["sequence"])
    x = torch.tensor([token2idx.get(t, 0) for t in tokens], dtype=torch.long).unsqueeze(
        -1
    )
    if len(tokens) > 1:
        src = torch.arange(len(tokens) - 1)
        dst = src + 1
        edge = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    else:
        edge = torch.zeros((2, 0), dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge,
        y=torch.tensor([label2idx[example["label"]]], dtype=torch.long),
    )
    data.seq = example["sequence"]
    return data


train_graphs = [seq_to_data(ex) for ex in spr["train"]]
dev_graphs = [seq_to_data(ex) for ex in spr["dev"]]
test_graphs = [seq_to_data(ex) for ex in spr["test"]]
batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ---------------------------------------------------------------------
# model ---------------------------------------------------------------
class SPR_GCN(nn.Module):
    def __init__(self, vocab, emb, ncls):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, emb, padding_idx=0)
        self.g1, self.g2 = GCNConv(emb, 64), GCNConv(64, 64)
        self.lin = nn.Linear(64, ncls)
        self.do = nn.Dropout(0.2)

    def forward(self, x, edge, batch):
        x = self.embed(x.squeeze(-1))
        x = self.g1(x, edge).relu()
        x = self.g2(x, edge).relu()
        x = global_mean_pool(x, batch)
        return self.lin(self.do(x))


# ---------------------------------------------------------------------
# evaluation ----------------------------------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss, preds, labels, seqs = 0, [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b.x, b.edge_index, b.batch)
            loss = criterion(out, b.y.view(-1))
            tot_loss += loss.item() * b.num_graphs
            pr = out.argmax(-1).cpu().tolist()
            lb = b.y.view(-1).cpu().tolist()
            preds += pr
            labels += lb
            seqs += b.seq
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    return tot_loss / len(loader.dataset), (cwa + swa) / 2, cwa, swa, preds, labels


# ---------------------------------------------------------------------
# hyper-parameter sweep ----------------------------------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
num_epochs = 5
experiment_data = {"weight_decay": {}}

for wd in weight_decays:
    tag = f"wd_{wd}"
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "weight_decay": wd,
    }
    model = SPR_GCN(len(token2idx), 32, len(label2idx)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for ep in range(1, num_epochs + 1):
        model.train()
        tloss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optim.step()
            tloss += loss.item() * batch.num_graphs
        tloss /= len(train_loader.dataset)
        vloss, vbwa, vcwa, vswa, *_ = evaluate(model, dev_loader)
        _, tbwa, _, _, *_ = evaluate(model, train_loader)
        exp["losses"]["train"].append(tloss)
        exp["losses"]["val"].append(vloss)
        exp["metrics"]["train"].append(tbwa)
        exp["metrics"]["val"].append(vbwa)
        exp["timestamps"].append(time.time())
        print(
            f"[{tag}] Ep{ep}: train_loss {tloss:.4f} val_loss {vloss:.4f} "
            f"BWA {vbwa:.4f} (CWA {vcwa:.4f}, SWA {vswa:.4f})"
        )

    # final test run
    tl, bwa, cwa, swa, pr, gt = evaluate(model, test_loader)
    print(f"[{tag}] Test -> loss {tl:.4f}  BWA {bwa:.4f}")
    exp["predictions"] = pr
    exp["ground_truth"] = gt
    experiment_data["weight_decay"][tag] = exp

    # optional plot per run
    e = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(e, exp["metrics"]["train"], label="Train BWA")
    plt.plot(e, exp["metrics"]["val"], label="Dev BWA")
    plt.title(f"BWA vs epochs ({tag})")
    plt.xlabel("Epoch")
    plt.ylabel("BWA")
    plt.legend()
    pth = os.path.join(working_dir, f"bwa_curve_{tag}.png")
    plt.tight_layout()
    plt.savefig(pth)
    plt.close()
    print("Curve saved to", pth)

# ---------------------------------------------------------------------
# persist -------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
