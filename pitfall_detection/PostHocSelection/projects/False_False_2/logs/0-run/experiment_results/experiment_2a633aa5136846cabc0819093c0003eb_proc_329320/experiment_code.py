import os, pathlib, random, string, warnings, sys, json
import numpy as np
import torch, math
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ---------- try to import benchmark helpers ----------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception:
    warnings.warn("Falling back to local metric impls")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError

    def _cnt_uni(xs, idx):
        return len(set(tok[idx] for tok in xs.strip().split() if len(tok) > idx))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_cnt_uni(s, 0) for s in seqs]
        return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / (
            sum(w) + 1e-9
        )

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [_cnt_uni(s, 1) for s in seqs]
        return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / (
            sum(w) + 1e-9
        )


# ---------- synthetic fallback ----------
def make_synth(n):
    shapes = list(string.ascii_uppercase[:6])
    cols = list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n):
        tokens = [
            random.choice(shapes) + random.choice(cols)
            for _ in range(random.randint(4, 9))
        ]
        seqs.append(" ".join(tokens))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ---------- load data ----------
root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root)
    train_seqs, train_labels = dsets["train"]["sequence"], dsets["train"]["label"]
    dev_seqs, dev_labels = dsets["dev"]["sequence"], dsets["dev"]["label"]
    test_seqs, test_labels = dsets["test"]["sequence"], dsets["test"]["label"]
    print("Loaded real SPR_BENCH.")
except Exception as e:
    warnings.warn(f"{e}\nGenerating synthetic data.")
    train = make_synth(512)
    dev = make_synth(128)
    test = make_synth(256)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

# ---------- vocab & featuriser ----------
shape_vocab = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
color_vocab = sorted(
    {tok[1] for seq in train_seqs for tok in seq.split() if len(tok) > 1}
)
shape2i, color2i = {s: i for i, s in enumerate(shape_vocab)}, {
    c: i for i, c in enumerate(color_vocab)
}
feat_dim = len(shape2i) + len(color2i)
print(f"Feature dim={feat_dim}")


def seq_to_feat(seq: str):
    sh = np.zeros(len(shape2i), np.float32)
    co = np.zeros(len(color2i), np.float32)
    for tok in seq.split():
        if tok:
            sh[shape2i.get(tok[0], 0)] += 1
            if len(tok) > 1:
                co[color2i.get(tok[1], 0)] += 1
    return np.concatenate([sh, co])


def encode(seqs, labels):
    X = np.stack([seq_to_feat(s) for s in seqs])
    y = np.asarray(labels, np.int64)
    return X, y


X_train, y_train = encode(train_seqs, train_labels)
X_dev, y_dev = encode(dev_seqs, dev_labels)
X_test, y_test = encode(test_seqs, test_labels)
n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"n_classes={n_classes}")

# ---------- torch loaders ----------
bs = 128
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=bs,
    shuffle=True,
)
dev_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev)), batch_size=bs
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=bs
)


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, dr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------- metric ----------
def compute_metrics(seqs, yt, yp):
    swa = shape_weighted_accuracy(seqs, yt, yp)
    cwa = color_weighted_accuracy(seqs, yt, yp)
    pha = 2 * swa * cwa / (swa + cwa + 1e-9)
    return swa, cwa, pha


# ---------- experiment container ----------
experiment_data = {"dropout_rate": {"spr_bench": {}}}

# ---------- hyper-parameter sweep ----------
rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
epochs = 10
best_dev_pha, best_rate, best_state = -1, None, None

for dr in rates:
    print(f"\n=== Training with dropout_rate={dr} ===")
    model = MLP(feat_dim, n_classes, dr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    hist = {
        "metrics": {"train_PHA": [], "dev_PHA": []},
        "losses": {"train": [], "dev": []},
        "epochs": [],
    }

    for ep in range(1, epochs + 1):
        # --- train ---
        model.train()
        tloss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tloss += loss.item() * xb.size(0)
        tloss /= len(train_loader.dataset)

        # --- dev ---
        model.eval()
        dloss, d_logits, d_y = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                lg = model(xb)
                dloss += crit(lg, yb).item() * xb.size(0)
                d_logits.append(lg.cpu())
                d_y.append(yb.cpu())
        dloss /= len(dev_loader.dataset)
        d_pred = torch.cat(d_logits).argmax(1).numpy()
        d_y = torch.cat(d_y).numpy()

        # PHA
        _, _, train_pha = compute_metrics(
            train_seqs,
            y_train,
            model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy(),
        )
        _, _, dev_pha = compute_metrics(dev_seqs, y_dev, d_pred)

        hist["epochs"].append(ep)
        hist["losses"]["train"].append(tloss)
        hist["losses"]["dev"].append(dloss)
        hist["metrics"]["train_PHA"].append(train_pha)
        hist["metrics"]["dev_PHA"].append(dev_pha)

        print(f"  Ep{ep:02d} tloss={tloss:.3f} dloss={dloss:.3f} dev_PHA={dev_pha:.3f}")

    # store history
    experiment_data["dropout_rate"]["spr_bench"][str(dr)] = hist

    # keep best
    if dev_pha > best_dev_pha:
        best_dev_pha, best_rate = dev_pha, dr
        best_state = model.state_dict()

    torch.cuda.empty_cache()

print(f"\nBest dropout_rate={best_rate} with dev_PHA={best_dev_pha:.4f}")

# ---------- test with best ----------
best_model = MLP(feat_dim, n_classes, best_rate).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_logits = []
with torch.no_grad():
    for xb, _ in test_loader:
        test_logits.append(best_model(xb.to(device)).cpu())
test_pred = torch.cat(test_logits).argmax(1).numpy()
swa, cwa, pha = compute_metrics(test_seqs, y_test, test_pred)
print(f"Test results  SWA={swa:.4f}  CWA={cwa:.4f}  PHA={pha:.4f}")

# store predictions / gt for best
best_key = experiment_data["dropout_rate"]["spr_bench"].setdefault("best", {})
best_key["rate"] = best_rate
best_key["test_metrics"] = {"SWA": swa, "CWA": cwa, "PHA": pha}
best_key["predictions"] = test_pred
best_key["ground_truth"] = y_test

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All done; artefacts written to ./working")
