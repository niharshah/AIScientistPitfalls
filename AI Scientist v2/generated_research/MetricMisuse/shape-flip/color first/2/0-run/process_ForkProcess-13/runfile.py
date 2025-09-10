import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_max_pool
from datasets import load_dataset, DatasetDict

# ---------------------- I/O & device ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- metrics ---------------------------
def count_colors(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shapes(seq):
    return len({tok[0] for tok in seq.split() if tok})


def CWA(seqs, y, p):
    w = [count_colors(s) for s in seqs]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0


def SWA(seqs, y, p):
    w = [count_shapes(s) for s in seqs]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0


def CpxWA(seqs, y, p):
    w = [count_colors(s) * count_shapes(s) for s in seqs]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0


# ---------------------- data loading ----------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def get_dataset():
    path = pathlib.Path(
        os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    try:
        ds = load_spr(path)
        print("Loaded SPR_BENCH from", path)
    except Exception as e:
        print("Falling back to tiny synthetic set:", e)
        shapes, colors = "ABC", "XYZ"

        def rand_seq():
            return " ".join(
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(3, 8))
            )

        def make(n):
            return {
                "id": list(range(n)),
                "sequence": [rand_seq() for _ in range(n)],
                "label": [random.randint(0, 2) for _ in range(n)],
            }

        from datasets import Dataset

        ds = DatasetDict(
            train=Dataset.from_dict(make(300)),
            dev=Dataset.from_dict(make(60)),
            test=Dataset.from_dict(make(60)),
        )
    return ds


dset = get_dataset()

# ---------------------- vocab -----------------------------
all_tokens = set(
    tok for split in dset.values() for seq in split["sequence"] for tok in seq.split()
)
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab = len(token2id) + 1
num_classes = len(set(dset["train"]["label"]))


# ---------------------- graph builder ---------------------
def seq_to_graph(seq, lbl):
    toks = seq.split()
    n = len(toks)
    x = torch.tensor([token2id[t] for t in toks], dtype=torch.long)

    edges = []
    rel = []
    # sequential edges
    for i in range(n - 1):
        edges += [[i, i + 1], [i + 1, i]]
        rel += [0, 0]
    # color edges
    color2idx = {}
    for i, t in enumerate(toks):
        color2idx.setdefault(t[1], []).append(i)
    for idxs in color2idx.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append([i, j])
                    rel.append(1)
    # shape edges
    shape2idx = {}
    for i, t in enumerate(toks):
        shape2idx.setdefault(t[0], []).append(i)
    for idxs in shape2idx.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append([i, j])
                    rel.append(2)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(rel, dtype=torch.long)
    return Data(
        x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([lbl]), seq=seq
    )


def build(split):
    return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_g, dev_g, test_g = build(dset["train"]), build(dset["dev"]), build(dset["test"])
train_loader = DataLoader(train_g, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_g, batch_size=128, shuffle=False)
test_loader = DataLoader(test_g, batch_size=128, shuffle=False)


# ---------------------- model -----------------------------
class RelGraphSAGE(nn.Module):
    def __init__(self, vocab, nclass, rel_emb_dim=16, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.rel_emb = nn.Embedding(3, rel_emb_dim)  # 3 relation types
        self.conv1 = SAGEConv(64 + rel_emb_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, nclass))

    def forward(self, data):
        x = self.emb(data.x)
        rel_e = self.rel_emb(data.edge_type)
        # concatenate relation embedding to source node feature
        src_feat = torch.cat([x[data.edge_index[0]], rel_e], dim=1)
        # message is (node_i, rel); conv expects size of src nodes
        out = self.conv1((x, src_feat), data.edge_index)
        out = torch.relu(out)
        out = self.conv2(out, data.edge_index)
        graph_emb = global_max_pool(out, data.batch)
        return self.lin(graph_emb)


model = RelGraphSAGE(vocab, num_classes).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------------------- tracking dict ---------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": [], "test": []},
        "predictions": [],
        "ground_truth": [],
    }
}


def evaluate(loader, split_name):
    model.eval()
    loss_tot = 0
    seqs = []
    preds = []
    gts = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss_tot += loss.item() * batch.num_graphs
            preds += out.argmax(1).cpu().tolist()
            gts += batch.y.view(-1).cpu().tolist()
            seqs += batch.seq
    loss_tot /= len(loader.dataset)
    cwa, swa, cpx = (
        CWA(seqs, gts, preds),
        SWA(seqs, gts, preds),
        CpxWA(seqs, gts, preds),
    )
    return loss_tot, (cwa, swa, cpx), preds, gts


epochs = 5
for ep in range(1, epochs + 1):
    # train
    model.train()
    tloss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch.num_graphs
    tloss /= len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((time.time(), tloss))

    # validate
    vloss, vmetrics, _, _ = evaluate(dev_loader, "val")
    experiment_data["SPR_BENCH"]["losses"]["val"].append((time.time(), vloss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((time.time(), vmetrics))
    print(
        f"Epoch {ep}: validation_loss = {vloss:.4f} | CWA {vmetrics[0]:.3f}  SWA {vmetrics[1]:.3f}  CpxWA {vmetrics[2]:.3f}"
    )

# test evaluation once training done
test_loss, tmetrics, tpreds, tgts = evaluate(test_loader, "test")
experiment_data["SPR_BENCH"]["losses"]["test"].append((time.time(), test_loss))
experiment_data["SPR_BENCH"]["metrics"]["test"].append((time.time(), tmetrics))
experiment_data["SPR_BENCH"]["predictions"] = tpreds
experiment_data["SPR_BENCH"]["ground_truth"] = tgts
print(
    f"TEST: loss {test_loss:.4f} | CWA {tmetrics[0]:.3f}  SWA {tmetrics[1]:.3f}  CpxWA {tmetrics[2]:.3f}"
)

# ---------------------- save ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved logs to", os.path.join(working_dir, "experiment_data.npy"))
