import os, pathlib, random, string, warnings, sys, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception:
    warnings.warn("Falling back to local metric definitions")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError

    def _count(seq, idx):
        return len(set(tok[idx] for tok in seq.strip().split() if len(tok) > idx))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count(s, 0) for s in seqs]
        return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
            sum(w) + 1e-9
        )

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count(s, 1) for s in seqs]
        return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
            sum(w) + 1e-9
        )


# ------------------------------------------------------------
def make_synthetic_dataset(n):
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


root_path = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root_path)
    train_seqs, train_labels = dsets["train"]["sequence"], dsets["train"]["label"]
    dev_seqs, dev_labels = dsets["dev"]["sequence"], dsets["dev"]["label"]
    test_seqs, test_labels = dsets["test"]["sequence"], dsets["test"]["label"]
    print("Loaded real SPR_BENCH.")
except Exception as e:
    warnings.warn(f"{e}\nGenerating synthetic data.")
    train = make_synthetic_dataset(512)
    dev = make_synthetic_dataset(128)
    test = make_synthetic_dataset(256)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

shape_vocab = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
color_vocab = sorted(
    {tok[1] for seq in train_seqs for tok in seq.split() if len(tok) > 1}
)
shape2idx = {s: i for i, s in enumerate(shape_vocab)}
color2idx = {c: i for i, c in enumerate(color_vocab)}
feat_dim = len(shape2idx) + len(color2idx)
print(f"Feature dim = {feat_dim}")


def seq_to_feature(seq: str) -> np.ndarray:
    sh = np.zeros(len(shape2idx), np.float32)
    co = np.zeros(len(color2idx), np.float32)
    for tok in seq.split():
        sh[shape2idx.get(tok[0], 0)] += 1.0
        if len(tok) > 1:
            co[color2idx.get(tok[1], 0)] += 1.0
    return np.concatenate([sh, co])


def encode(seqs, labels):
    return np.stack([seq_to_feature(s) for s in seqs]), np.array(labels, np.int64)


X_train, y_train = encode(train_seqs, train_labels)
X_dev, y_dev = encode(dev_seqs, dev_labels)
X_test, y_test = encode(test_seqs, test_labels)

n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"Classes: {n_classes}")

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


def compute_metrics(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    pha = 2 * swa * cwa / (swa + cwa + 1e-9)
    return swa, cwa, pha


class MLP(nn.Module):
    def __init__(self, in_dim, hid, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, n_cls)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
experiment_data = {"hidden_dim_tuning": {}}  # mandatory key

hidden_dims = [64, 128, 256, 512, 1024]
epochs = 10

for hd in hidden_dims:
    print(f"\n--- Training with hidden_dim={hd} ---")
    model = MLP(feat_dim, hd, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    exp = {
        "metrics": {"train_PHA": [], "dev_PHA": []},
        "losses": {"train": [], "dev": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "epochs": [],
    }

    for ep in range(1, epochs + 1):
        # train
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

        # dev
        model.eval()
        dloss, d_logits, d_y = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                dloss += crit(logits, yb).item() * xb.size(0)
                d_logits.append(logits.cpu())
                d_y.append(yb.cpu())
        dloss /= len(dev_loader.dataset)
        d_pred = torch.cat(d_logits).argmax(1).numpy()
        d_y = torch.cat(d_y).numpy()

        # metrics
        _, _, tr_pha = compute_metrics(
            train_seqs,
            y_train,
            model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy(),
        )
        _, _, dv_pha = compute_metrics(dev_seqs, y_dev, d_pred)

        exp["epochs"].append(ep)
        exp["losses"]["train"].append(tloss)
        exp["losses"]["dev"].append(dloss)
        exp["metrics"]["train_PHA"].append(tr_pha)
        exp["metrics"]["dev_PHA"].append(dv_pha)

        print(
            f"hid={hd}  Epoch {ep}: train_loss={tloss:.4f} dev_loss={dloss:.4f} dev_PHA={dv_pha:.4f}"
        )

    # final test evaluation
    model.eval()
    with torch.no_grad():
        t_logits = []
        for xb, _ in test_loader:
            t_logits.append(model(xb.to(device)).cpu())
    t_pred = torch.cat(t_logits).argmax(1).numpy()
    swa, cwa, pha = compute_metrics(test_seqs, y_test, t_pred)
    exp["predictions"] = t_pred.tolist()
    exp["test_metrics"] = {"SWA": swa, "CWA": cwa, "PHA": pha}
    print(f"hid={hd} TEST  SWA={swa:.4f} CWA={cwa:.4f} PHA={pha:.4f}")

    experiment_data["hidden_dim_tuning"][str(hd)] = exp

# ------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")

print("All done; artefacts written to ./working")
