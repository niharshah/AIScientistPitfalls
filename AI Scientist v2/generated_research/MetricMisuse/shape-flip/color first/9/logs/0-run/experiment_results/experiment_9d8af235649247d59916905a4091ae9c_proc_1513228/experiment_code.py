import os, random, math, time, pathlib, itertools, warnings
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

warnings.filterwarnings("ignore")
# ---------- housekeeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
print(f"Using device: {device}")

# ---------- experiment container ----------
experiment_data = {"num_epochs": {"SPR": {}}}


# ---------- metric helpers ----------
def _uniq_shapes(seq):
    return len(set(tok[0] for tok in seq.split()))


def _uniq_colors(seq):
    return len(set(tok[1] for tok in seq.split()))


def sdwa_metric(seqs, y_true, y_pred):
    w = [_uniq_shapes(s) + _uniq_colors(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / sum(w)


# ---------- data loading ----------
def try_load_real():
    try:
        from SPR import load_spr_bench

        dset = load_spr_bench(pathlib.Path("./SPR_BENCH"))
        return dset
    except Exception as e:
        print("Real dataset not found, generating synthetic:", e)
        return None


def gen_synth(N):
    shapes, colors = list("ABCD"), list("1234")
    seqs, labels = [], []
    for _ in range(N):
        L = random.randint(3, 10)
        toks = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
        seq = " ".join(toks)
        labels.append((_uniq_shapes(seq) * _uniq_colors(seq)) % 4)
        seqs.append(seq)
    return {"sequence": seqs, "label": labels}


real = try_load_real()
if real:
    train_raw, dev_raw, test_raw = real["train"], real["dev"], real["test"]
else:
    train_raw, dev_raw, test_raw = gen_synth(2000), gen_synth(500), gen_synth(500)


# ---------- vocab ----------
def build_vocabs(sets):
    sh, co = set(), set()
    for rs in sets:
        for seq in rs["sequence"]:
            for tok in seq.split():
                sh.add(tok[0])
                co.add(tok[1])
    return {s: i for i, s in enumerate(sorted(sh))}, {
        c: i for i, c in enumerate(sorted(co))
    }


shape_vocab, color_vocab = build_vocabs([train_raw, dev_raw, test_raw])


# ---------- graphs ----------
def seq_to_graph(seq, label):
    toks = seq.split()
    n = len(toks)
    sids = [shape_vocab[t[0]] for t in toks]
    cids = [color_vocab[t[1]] for t in toks]
    pos = [i / (n - 1 if n > 1 else 1) for i in range(n)]
    x = torch.tensor(
        [[si, ci, po] for si, ci, po in zip(sids, cids, pos)], dtype=torch.float
    )
    edges = [[i, i + 1] for i in range(n - 1)]
    edge_index = torch.tensor(
        list(itertools.chain(edges, [(j, i) for i, j in edges])), dtype=torch.long
    ).t()
    return Data(
        x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), seq=seq
    )


def build_ds(raw):
    return [seq_to_graph(s, l) for s, l in zip(raw["sequence"], raw["label"])]


train_ds, dev_ds, test_ds = build_ds(train_raw), build_ds(dev_raw), build_ds(test_raw)


# ---------- model ----------
class SPRGCN(nn.Module):
    def __init__(self, in_dim, hid, n_cls):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hid), GCNConv(hid, hid)
        self.lin = nn.Linear(hid, n_cls)

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = self.conv1(x, ei).relu()
        x = self.conv2(x, ei).relu()
        return self.lin(global_mean_pool(x, b))


# ---------- train helpers ----------
def run_epoch(model, loader, optim, criterion, train=True):
    (model.train() if train else model.eval())
    tot_loss, y_t, y_p, seqs = 0, [], [], []
    for d in loader:
        d = d.to(device)
        optim.zero_grad()
        out = model(d)
        loss = criterion(out, d.y.view(-1))
        if train:
            loss.backward()
            optim.step()
        tot_loss += loss.item() * d.num_graphs
        p = out.argmax(1).cpu().tolist()
        y_p.extend(p)
        y_t.extend(d.y.cpu().tolist())
        seqs.extend(d.seq)
    return tot_loss / len(loader.dataset), sdwa_metric(seqs, y_t, y_p), y_t, y_p


# ---------- hyperparameter tuning ----------
search_space = [10, 30, 50]
batch_train, batch_val, batch_test = 32, 64, 64
train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
val_loader = DataLoader(dev_ds, batch_size=batch_val)
test_loader = DataLoader(test_ds, batch_size=batch_test)
for num_epochs in search_space:
    model = SPRGCN(3, 64, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # containers
    cont = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "best_val": None,
    }
    patience, pat_cnt = 5, 0
    best_val = float("inf")
    for ep in range(1, num_epochs + 1):
        tr_loss, tr_sdwa, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, True
        )
        vl_loss, vl_sdwa, _, _ = run_epoch(
            model, val_loader, optimizer, criterion, False
        )
        cont["epochs"].append(ep)
        cont["losses"]["train"].append(tr_loss)
        cont["losses"]["val"].append(vl_loss)
        cont["metrics"]["train"].append(tr_sdwa)
        cont["metrics"]["val"].append(vl_sdwa)
        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            pat_cnt = 0
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            print(f"Early stop @ epoch {ep} for num_epochs={num_epochs}")
            break
    # test
    _, test_sdwa, gt, pr = run_epoch(model, test_loader, optimizer, criterion, False)
    cont["predictions"], cont["ground_truth"] = pr, gt
    cont["best_val"] = best_val
    experiment_data["num_epochs"]["SPR"][str(num_epochs)] = cont
    print(f"[{num_epochs}] Test SDWA: {test_sdwa:.4f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
# optional: plot one curve per setting
for k, v in experiment_data["num_epochs"]["SPR"].items():
    plt.figure()
    plt.plot(v["epochs"], v["losses"]["train"], label="train")
    plt.plot(v["epochs"], v["losses"]["val"], label="val")
    plt.title(f"Loss (num_epochs={k})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{k}.png"), dpi=150)
    plt.close()
