import os, random, itertools, pathlib

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np, torch, math
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, GlobalAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------- experiment container ---------------------------- #
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "best_epoch": None,
    }
}


# ---------------------------- metric helpers ----------------------------------#
def _uniq_colors(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _uniq_shapes(seq):
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y_true, y_pred):
    w = [_uniq_colors(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def swa(seqs, y_true, y_pred):
    w = [_uniq_shapes(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def hwa(c, s, eps=1e-9):
    return 2 * c * s / (c + s + eps)


# ---------------------------- load data ---------------------------------------#
def try_load_real():
    try:
        from SPR import load_spr_bench

        root = pathlib.Path("./SPR_BENCH")
        dset = load_spr_bench(root)
        return dset["train"], dset["dev"], dset["test"]
    except Exception as e:
        print("Falling back to synthetic data because:", e)
        return None


def synth_split(n):
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(n):
        ln = random.randint(3, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
        seq = " ".join(toks)
        # simple synthetic rule
        label = (len(set(t[0] for t in toks)) * len(set(t[1] for t in toks))) % 4
        seqs.append(seq)
        labels.append(label)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real
else:
    train_raw, dev_raw, test_raw = synth_split(2500), synth_split(600), synth_split(600)


# ---------------------------- vocab -------------------------------------------#
def build_voc(*splits):
    sv, cv = set(), set()
    for split in splits:
        for seq in split["sequence"]:
            for t in seq.split():
                sv.add(t[0])
                cv.add(t[1])
    return {s: i for i, s in enumerate(sorted(sv))}, {
        c: i for i, c in enumerate(sorted(cv))
    }


shape_vocab, color_vocab = build_voc(train_raw, dev_raw, test_raw)
S, C = len(shape_vocab), len(color_vocab)


# -------------------------- seq -> graph --------------------------------------#
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    sid = [shape_vocab[t[0]] for t in toks]
    cid = [color_vocab[t[1]] for t in toks]
    pos = [i / (n - 1 if n > 1 else 1) for i in range(n)]

    sh_oh = torch.nn.functional.one_hot(torch.tensor(sid), num_classes=S).float()
    co_oh = torch.nn.functional.one_hot(torch.tensor(cid), num_classes=C).float()
    pos_feat = torch.tensor(pos, dtype=torch.float32).unsqueeze(1)
    x = torch.cat([sh_oh, co_oh, pos_feat], 1)

    edge_index = []
    edge_type = []
    # 0: sequential
    for i in range(n - 1):
        for a, b in [(i, i + 1), (i + 1, i)]:
            edge_index.append([a, b])
            edge_type.append(0)
    # 1: same shape
    for s in set(sid):
        idx = [i for i, v in enumerate(sid) if v == s]
        for a, b in itertools.permutations(idx, 2):
            edge_index.append([a, b])
            edge_type.append(1)
    # 2: same color
    for c_ in set(cid):
        idx = [i for i, v in enumerate(cid) if v == c_]
        for a, b in itertools.permutations(idx, 2):
            edge_index.append([a, b])
            edge_type.append(2)

    if not edge_index:
        edge_index = [[0, 0]]
        edge_type = [0]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor(int(label)),
        seq=seq,
    )


def to_pyg(split):
    if hasattr(split, "__getitem__") and not isinstance(split, dict):
        return [seq_to_graph(ex["sequence"], int(ex["label"])) for ex in split]
    else:
        return [seq_to_graph(s, l) for s, l in zip(split["sequence"], split["label"])]


train_ds, dev_ds, test_ds = map(to_pyg, (train_raw, dev_raw, test_raw))
num_classes = len({d.y.item() for d in train_ds + dev_ds + test_ds})


# -------------------------- model ---------------------------------------------#
class SPR_RGCN(nn.Module):
    def __init__(self, in_dim, hid, out, rel):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hid, rel, num_bases=None)
        self.conv2 = RGCNConv(hid, hid, rel, num_bases=None)
        self.pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hid, 1)))
        self.fc = nn.Linear(hid, out)

    def forward(self, data):
        x, ei, et, b = data.x, data.edge_index, data.edge_type, data.batch
        x = self.conv1(x, ei, et).relu()
        x = self.conv2(x, ei, et).relu()
        x = self.pool(x, b)
        return self.fc(x)


model = SPR_RGCN(S + C + 1, 64, num_classes, rel=3).to(device)


# -------------------------- training utils ------------------------------------#
def run_epoch(model, loader, criterion, opt=None):
    model.train() if opt else model.eval()
    total_loss = 0
    seqs = []
    ys = []
    ps = []
    for batch in loader:
        batch = batch.to(device)
        if opt:
            opt.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        if opt:
            loss.backward()
            opt.step()
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(1).detach().cpu().tolist()
        labels = batch.y.cpu().tolist()
        seqs.extend(batch.seq)
        ys.extend(labels)
        ps.extend(preds)
    avg_loss = total_loss / len(loader.dataset)
    c, s = cwa(seqs, ys, ps), swa(seqs, ys, ps)
    return avg_loss, {"CWA": c, "SWA": s, "HWA": hwa(c, s)}, ys, ps


# -------------------------- training loop -------------------------------------#
BATCH = 32
EPOCHS = 20
LR = 5e-4
WD = 1e-4
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(dev_ds, batch_size=BATCH * 2, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

best_hwa = -1
best_state = None
best_epoch = 0
for epoch in range(1, EPOCHS + 1):
    tloss, tmet, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    vloss, vmet, _, _ = run_epoch(model, val_loader, criterion)
    print(f'Epoch {epoch}: validation_loss = {vloss:.4f}, HWA = {vmet["HWA"]:.4f}')
    experiment_data["SPR"]["epochs"].append(epoch)
    experiment_data["SPR"]["losses"]["train"].append(tloss)
    experiment_data["SPR"]["losses"]["val"].append(vloss)
    experiment_data["SPR"]["metrics"]["train"].append(tmet)
    experiment_data["SPR"]["metrics"]["val"].append(vmet)
    if vmet["HWA"] > best_hwa:
        best_hwa = vmet["HWA"]
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_epoch = epoch
        print(f"  New best model at epoch {epoch} with HWA {best_hwa:.4f}")

# -------------------------- test ------------------------------------------------#
model.load_state_dict(best_state)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
_, test_met, gt, pred = run_epoch(model, test_loader, criterion)
print(
    f'Test CWA={test_met["CWA"]:.3f}, SWA={test_met["SWA"]:.3f}, HWA={test_met["HWA"]:.3f}'
)

experiment_data["SPR"]["predictions"] = pred
experiment_data["SPR"]["ground_truth"] = gt
experiment_data["SPR"]["best_epoch"] = best_epoch

# -------------------------- save artefacts -------------------------------------#
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
