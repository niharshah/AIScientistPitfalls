import os, random, string, time, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datasets import load_dataset, Dataset, DatasetDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ------------------------ bookkeeping dict ------------------------
experiment_data = {"weight_decay": {}}
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ------------------------ helpers ---------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def complexity_weight(sequence: str) -> int:
    return count_color_variety(sequence) + count_shape_variety(sequence)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    return float(sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p)) / max(
        1, sum(w)
    )


# ------------------------ data ------------------------------------
SPR_PATH = os.environ.get("SPR_BENCH_PATH", "./SPR_BENCH")


def load_spr(path: str) -> DatasetDict:
    if os.path.isdir(path):

        def _csv(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(path, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        return DatasetDict(train=_csv("train"), dev=_csv("dev"), test=_csv("test"))
    shapes = list(string.ascii_uppercase[:6])
    colors = list(range(6))

    def make_seq():
        return " ".join(
            random.choice(shapes) + str(random.choice(colors))
            for _ in range(random.randint(4, 9))
        )

    def label_rule(seq):
        return sum(int(tok[1]) for tok in seq.split()) % 2

    def build(n):
        seqs = [make_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [label_rule(s) for s in seqs],
            }
        )

    return DatasetDict(train=build(800), dev=build(200), test=build(200))


spr = load_spr(SPR_PATH)
num_classes = len(set(spr["train"]["label"]))

# vocab
vocab = {}
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)


def seq_to_graph(seq, label):
    ids = [vocab[t] for t in seq.split()]
    edge = []
    for i in range(len(ids) - 1):
        edge.append([i, i + 1])
        edge.append([i + 1, i])
    edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
    return Data(
        x=torch.tensor(ids, dtype=torch.long),
        edge_index=edge,
        y=torch.tensor([label], dtype=torch.long),
        seq=seq,
    )


def encode(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_graphs, dev_graphs, test_graphs = map(
    encode, [spr["train"], spr["dev"], spr["test"]]
)
batch_size = 128
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)


# ------------------------ model -----------------------------------
class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.c1 = SAGEConv(emb_dim, hid)
        self.c2 = SAGEConv(hid, hid)
        self.lin = nn.Linear(hid, num_classes)

    def forward(self, data):
        x = self.emb(data.x)
        x = F.relu(self.c1(x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = tot_corr = tot = 0
    seqs_all, preds_all, ys_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch.num_graphs
        preds = out.argmax(-1).cpu().tolist()
        ys = batch.y.cpu().tolist()
        tot_corr += sum(int(p == y) for p, y in zip(preds, ys))
        tot += batch.num_graphs
        seqs_all += batch.seq
        preds_all += preds
        ys_all += ys
    avg_loss = tot_loss / tot
    acc = tot_corr / tot
    cowa = complexity_weighted_accuracy(seqs_all, ys_all, preds_all)
    return avg_loss, acc, cowa, preds_all, ys_all, seqs_all


# -------------------- hyper-parameter grid search -----------------
weight_decays = [0, 1e-4, 1e-3, 1e-2]
EPOCHS = 5

for wd in weight_decays:
    tag = f"wd_{wd}"
    print(f"\n=== training with weight_decay={wd} ===")
    exp_entry = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
    model = GNNClassifier(len(vocab), num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_cowa, *_ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, val_cowa, *_ = run_epoch(model, dev_loader, criterion)
        exp_entry["losses"]["train"].append(tr_loss)
        exp_entry["losses"]["val"].append(val_loss)
        exp_entry["metrics"]["train"].append({"acc": tr_acc, "cowa": tr_cowa})
        exp_entry["metrics"]["val"].append({"acc": val_acc, "cowa": val_cowa})
        print(
            f"Epoch {epoch}: wd={wd} val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_cowa={val_cowa:.3f}"
        )
    # test after training
    test_loss, test_acc, test_cowa, preds, gt, seqs = run_epoch(
        model, test_loader, criterion
    )
    print(
        f"TEST wd={wd} -- loss:{test_loss:.4f} acc:{test_acc:.3f} cowa:{test_cowa:.3f}"
    )
    exp_entry["predictions"] = preds
    exp_entry["ground_truth"] = gt
    exp_entry["sequences"] = seqs
    experiment_data["weight_decay"][tag] = exp_entry

# ------------------------ save ------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
