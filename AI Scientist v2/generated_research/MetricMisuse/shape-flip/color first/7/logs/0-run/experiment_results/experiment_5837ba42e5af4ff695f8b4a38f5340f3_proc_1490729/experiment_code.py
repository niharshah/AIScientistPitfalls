import os, random, string, time, json
import numpy as np
import torch, math
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool


# ----------------------- misc helpers -----------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- SPR helpers ----------
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split()))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def complexity_weight(s):
    return count_color_variety(s) + count_shape_variety(s)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    return float(sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p)) / max(
        1, sum(w)
    )


# ---------- dataset (real or synthetic) ----------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path):
    if os.path.isdir(path):
        print("Loading real SPR_BENCH from", path)

        def _csv(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict(train=_csv("train"), dev=_csv("dev"), test=_csv("test"))
    # synthetic fallback
    print("No SPR_BENCH found, generating toy data")
    shapes = list(string.ascii_uppercase[:6])
    colors = list(range(6))

    def make_seq():
        L = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + str(random.choice(colors)) for _ in range(L)
        )

    def lab_rule(seq):
        return sum(int(tok[1]) for tok in seq.split()) % 2

    def build(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [lab_rule(s) for s in seqs],
            }
        )

    return DatasetDict(train=build(800), dev=build(200), test=build(200))


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))
print({k: len(v) for k, v in spr.items()}, "classes:", num_classes)

# ---------- vocab & graph encoding ----------
vocab = {}


def add_tok(t):
    if t not in vocab:
        vocab[t] = len(vocab)


for s in spr["train"]["sequence"]:
    for t in s.split():
        add_tok(t)


def seq_to_graph(seq, label):
    idx = [vocab[t] for t in seq.split()]
    edge = []
    for i in range(len(idx) - 1):
        edge.append([i, i + 1])
        edge.append([i + 1, i])
    edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(idx), edge_index=edge, y=torch.tensor([label]), seq=seq)


def encode(ds):
    return [seq_to_graph(s, l) for s, l in zip(ds["sequence"], ds["label"])]


train_graphs, dev_graphs, test_graphs = (
    encode(spr["train"]),
    encode(spr["dev"]),
    encode(spr["test"]),
)


# ---------- model ----------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=64, num_cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.c1 = SAGEConv(emb, hid)
        self.c2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, num_cls)

    def forward(self, data):
        x = self.emb(data.x)
        x = F.relu(self.c1(x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


# ---------- train / eval ----------
def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tloss = tcorrect = tsamp = 0
    seq_all, pred_all, lab_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tloss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        labs = batch.y.cpu().tolist()
        tcorrect += sum(int(p == y) for p, y in zip(preds, labs))
        tsamp += batch.num_graphs
        seq_all.extend(batch.seq)
        pred_all.extend(preds)
        lab_all.extend(labs)
    avg = tloss / tsamp
    acc = tcorrect / tsamp
    cowa = complexity_weighted_accuracy(seq_all, lab_all, pred_all)
    return avg, acc, cowa, pred_all, lab_all, seq_all


# ---------- hyper-parameter tuning : batch size ----------
batch_grid = [32, 64, 128, 256]  # explore
EPOCHS = 5
experiment_data = {"batch_size_tuning": {"SPR_BENCH": {}}}

for bs in batch_grid:
    print(f"\n=== Training with batch_size={bs} ===")
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=bs, shuffle=False)

    model = GNNClassifier(len(vocab), num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=1e-3)

    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_cowa, *_ = run_epoch(model, train_loader, criterion, optim)
        vl_loss, vl_acc, vl_cowa, *_ = run_epoch(model, dev_loader, criterion)
        log["losses"]["train"].append(tr_loss)
        log["losses"]["val"].append(vl_loss)
        log["metrics"]["train"].append({"acc": tr_acc, "cowa": tr_cowa})
        log["metrics"]["val"].append({"acc": vl_acc, "cowa": vl_cowa})
        print(
            f"Ep{epoch}: val_loss={vl_loss:.4f}, acc={vl_acc:.3f}, cowa={vl_cowa:.3f}"
        )

    # final test
    te_loss, te_acc, te_cowa, preds, gts, seqs = run_epoch(
        model, test_loader, criterion
    )
    print(f"TEST: loss={te_loss:.4f}, acc={te_acc:.3f}, cowa={te_cowa:.3f}")
    log["test"] = {"loss": te_loss, "acc": te_acc, "cowa": te_cowa}
    log["predictions"] = preds
    log["ground_truth"] = gts
    log["sequences"] = seqs

    experiment_data["batch_size_tuning"]["SPR_BENCH"][str(bs)] = log

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
