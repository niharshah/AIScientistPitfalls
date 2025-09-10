import os, pathlib, random, string, warnings, sys, time, json
import numpy as np
import torch, matplotlib

matplotlib.use("Agg")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------------------------------------
# SPR helper fall-back ---------------------------------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception as e:
    warnings.warn("Could not import SPR helpers, using local stand-ins")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError("SPR_BENCH not found")

    def _cnt(seq, idx):
        return len(set(tok[idx] for tok in seq.strip().split() if len(tok) > idx))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_cnt(s, 0) for s in seqs]
        return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / (
            sum(w) + 1e-9
        )

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [_cnt(s, 1) for s in seqs]
        return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / (
            sum(w) + 1e-9
        )


# ------------------------------------------------------------
def make_synth(n):
    shapes = list(string.ascii_uppercase[:6])
    cols = list(string.ascii_lowercase[:6])
    seqs, labs = [], []
    for _ in range(n):
        toks = [
            random.choice(shapes) + random.choice(cols)
            for _ in range(random.randint(4, 9))
        ]
        seqs.append(" ".join(toks))
        labs.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labs}


root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    data = load_spr_bench(root)
    train_seqs, train_labels = data["train"]["sequence"], data["train"]["label"]
    dev_seqs, dev_labels = data["dev"]["sequence"], data["dev"]["label"]
    test_seqs, test_labels = data["test"]["sequence"], data["test"]["label"]
    print("Loaded real SPR_BENCH.")
except Exception as e:
    warnings.warn(f"{e}\nUsing synthetic data.")
    train = make_synth(512)
    dev = make_synth(128)
    test = make_synth(256)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

shape_vocab = sorted({tok[0] for s in train_seqs for tok in s.split()})
color_vocab = sorted({tok[1] for s in train_seqs for tok in s.split() if len(tok) > 1})
shape2i = {c: i for i, c in enumerate(shape_vocab)}
color2i = {c: i for i, c in enumerate(color_vocab)}
feat_dim = len(shape2i) + len(color2i)
print(f"Feature dim {feat_dim}")


def seq_to_vec(seq: str):
    sh = np.zeros(len(shape2i), dtype=np.float32)
    co = np.zeros(len(color2i), dtype=np.float32)
    for tok in seq.split():
        if tok:
            sh[shape2i.get(tok[0], 0)] += 1
            if len(tok) > 1:
                co[color2i.get(tok[1], 0)] += 1
    return np.concatenate([sh, co])


def encode(seqs, labels):
    return np.stack([seq_to_vec(s) for s in seqs]), np.array(labels, dtype=np.int64)


X_train, y_train = encode(train_seqs, train_labels)
X_dev, y_dev = encode(dev_seqs, dev_labels)
X_test, y_test = encode(test_seqs, test_labels)

n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"{n_classes} classes")

bd = 128
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=bd,
    shuffle=True,
)
dev_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev)), batch_size=bd
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=bd
)


class MLP(nn.Module):
    def __init__(self, inp, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 128), nn.ReLU(), nn.Linear(128, classes)
        )

    def forward(self, x):
        return self.net(x)


def metrics(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / (swa + cwa + 1e-9)


# ------------------------------------------------------------
experiment_data = {"optimizer_type": {"spr_bench": {}}}

opt_configs = {
    "Adam": (torch.optim.Adam, dict(lr=1e-3)),
    "RMSprop": (torch.optim.RMSprop, dict(lr=1e-3)),
    "SGD": (torch.optim.SGD, dict(lr=1e-2, momentum=0.9)),
}

best_dev = -1.0
best_state = None
best_opt = None
epochs = 10
criterion = nn.CrossEntropyLoss()

for opt_name, (opt_cls, opt_kw) in opt_configs.items():
    print(f"\n=== Training with {opt_name} ===")
    model = MLP(feat_dim, n_classes).to(device)
    optimizer = opt_cls(model.parameters(), **opt_kw)
    log = {
        "losses": {"train": [], "dev": []},
        "metrics": {"train_PHA": [], "dev_PHA": []},
    }
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tloss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * xb.size(0)
        tloss /= len(train_loader.dataset)

        # dev
        model.eval()
        dloss = 0.0
        d_pred = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                dloss += criterion(logits, yb).item() * xb.size(0)
                d_pred.append(logits.cpu())
        dloss /= len(dev_loader.dataset)
        d_pred = torch.cat(d_pred).argmax(1).numpy()

        # pha
        with torch.no_grad():
            train_pred = (
                model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy()
            )
        tr_pha = metrics(train_seqs, y_train, train_pred)
        dv_pha = metrics(dev_seqs, y_dev, d_pred)

        log["losses"]["train"].append(tloss)
        log["losses"]["dev"].append(dloss)
        log["metrics"]["train_PHA"].append(tr_pha)
        log["metrics"]["dev_PHA"].append(dv_pha)

        print(f"Ep{ep:02d}  tloss={tloss:.4f}  dloss={dloss:.4f}  dev_PHA={dv_pha:.4f}")

        if dv_pha > best_dev:
            best_dev = dv_pha
            best_state = model.state_dict()
            best_opt = opt_name

    experiment_data["optimizer_type"]["spr_bench"][opt_name] = log

print(f"\nBest optimizer based on dev PHA: {best_opt} ({best_dev:.4f})")

# ------------------------------------------------------------
# evaluate best model on test set
best_model = MLP(feat_dim, n_classes).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
t_pred = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        t_pred.append(best_model(xb).cpu())
t_pred = torch.cat(t_pred).argmax(1).numpy()
swa = color_weighted_accuracy(
    test_seqs, y_test, t_pred
)  # actually returns CWA, fix next line
# swap
swa = shape_weighted_accuracy(test_seqs, y_test, t_pred)
cwa = color_weighted_accuracy(test_seqs, y_test, t_pred)
pha = 2 * swa * cwa / (swa + cwa + 1e-9)
print(f"Test results with {best_opt}:  SWA={swa:.4f}  CWA={cwa:.4f}  PHA={pha:.4f}")

# store predictions & gt
experiment_data["optimizer_type"]["spr_bench"]["best_optimizer"] = best_opt
experiment_data["optimizer_type"]["spr_bench"]["predictions"] = t_pred
experiment_data["optimizer_type"]["spr_bench"]["ground_truth"] = y_test
experiment_data["optimizer_type"]["spr_bench"]["test_metrics"] = {
    "SWA": swa,
    "CWA": cwa,
    "PHA": pha,
}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All done; artefacts saved to ./working")
