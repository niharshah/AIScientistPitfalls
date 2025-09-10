import os, pathlib, random, string, warnings, sys, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# basic GPU / CPU handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------------------------------------
# try to import official helper; otherwise define fall-backs
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception as e:
    warnings.warn("Could not import SPR helpers, falling back to local defs")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError("SPR_BENCH not found")

    def count_shape_variety(sequence: str):
        return len(set(tok[0] for tok in sequence.strip().split() if tok))

    def count_color_variety(sequence: str):
        return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [count_shape_variety(s) for s in seqs]
        c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(c) / (sum(w) + 1e-9)

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [count_color_variety(s) for s in seqs]
        c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(c) / (sum(w) + 1e-9)


# ------------------------------------------------------------
# helper to fabricate synthetic data if benchmark missing
def make_synthetic_dataset(n_rows):
    shapes = list(string.ascii_uppercase[:6])  # A–F
    cols = list(string.ascii_lowercase[:6])  # a–f
    seqs, labels = [], []
    for _ in range(n_rows):
        length = random.randint(4, 9)
        tokens = [random.choice(shapes) + random.choice(cols) for _ in range(length)]
        seqs.append(" ".join(tokens))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ------------------------------------------------------------
# load data (real or synthetic)
root_path = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root_path)
    print("Loaded real SPR_BENCH.")
    train_data = dsets["train"]
    dev_data = dsets["dev"]
    test_data = dsets["test"]
    train_seqs, train_labels = train_data["sequence"], train_data["label"]
    dev_seqs, dev_labels = dev_data["sequence"], dev_data["label"]
    test_seqs, test_labels = test_data["sequence"], test_data["label"]
except Exception as e:
    warnings.warn(f"{e}\nGenerating synthetic data instead.")
    train = make_synthetic_dataset(512)
    dev = make_synthetic_dataset(128)
    test = make_synthetic_dataset(256)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

# ------------------------------------------------------------
# build vocabularies of shapes and colours
shape_vocab = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
color_vocab = sorted(
    {tok[1] for seq in train_seqs for tok in seq.split() if len(tok) > 1}
)
shape2idx = {s: i for i, s in enumerate(shape_vocab)}
color2idx = {c: i for i, c in enumerate(color_vocab)}
feat_dim = len(shape2idx) + len(color2idx)
print(f"Feature dim = {feat_dim} (|S|={len(shape2idx)}, |C|={len(color2idx)})")


def seq_to_feature(seq: str) -> np.ndarray:
    sh = np.zeros(len(shape2idx), dtype=np.float32)
    co = np.zeros(len(color2idx), dtype=np.float32)
    for tok in seq.split():
        if tok:
            sh[shape2idx.get(tok[0], 0)] += 1.0
            if len(tok) > 1:
                co[color2idx.get(tok[1], 0)] += 1.0
    return np.concatenate([sh, co])


# ------------------------------------------------------------
# encode datasets
def encode_dataset(seqs, labels):
    X = np.stack([seq_to_feature(s) for s in seqs])
    y = np.array(labels, dtype=np.int64)
    return X, y


X_train, y_train = encode_dataset(train_seqs, train_labels)
X_dev, y_dev = encode_dataset(dev_seqs, dev_labels)
X_test, y_test = encode_dataset(test_seqs, test_labels)

n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"Detected {n_classes} classes")

# ------------------------------------------------------------
# torch datasets / loaders
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


# ------------------------------------------------------------
# simple MLP model
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(feat_dim, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------
# experiment tracking
experiment_data = {
    "spr_bench": {
        "metrics": {"train_PHA": [], "dev_PHA": []},
        "losses": {"train": [], "dev": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


def compute_metrics(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    pha = 2 * swa * cwa / (swa + cwa + 1e-9)
    return swa, cwa, pha


# ------------------------------------------------------------
# training loop
epochs = 10
for epoch in range(1, epochs + 1):
    # ---- training ----
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    train_loss = epoch_loss / len(train_loader.dataset)

    # ---- validation ----
    model.eval()
    with torch.no_grad():
        dev_logits, dev_ys = [], []
        epoch_loss = 0.0
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            epoch_loss += loss.item() * xb.size(0)
            dev_logits.append(logits.cpu())
            dev_ys.append(yb.cpu())
        dev_loss = epoch_loss / len(dev_loader.dataset)
        dev_logits = torch.cat(dev_logits).argmax(1).numpy()
        dev_ys = torch.cat(dev_ys).numpy()

    # compute PHA
    _, _, train_pha = compute_metrics(
        train_seqs,
        y_train,
        model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy(),
    )
    _, _, dev_pha = compute_metrics(dev_seqs, y_dev, dev_logits)

    # logging
    experiment_data["spr_bench"]["epochs"].append(epoch)
    experiment_data["spr_bench"]["losses"]["train"].append(train_loss)
    experiment_data["spr_bench"]["losses"]["dev"].append(dev_loss)
    experiment_data["spr_bench"]["metrics"]["train_PHA"].append(train_pha)
    experiment_data["spr_bench"]["metrics"]["dev_PHA"].append(dev_pha)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}  dev_PHA={dev_pha:.4f}"
    )

# ------------------------------------------------------------
# final evaluation on test set
model.eval()
with torch.no_grad():
    test_logits = []
    for xb, _ in test_loader:
        xb = xb.to(device)
        test_logits.append(model(xb).cpu())
    test_pred = torch.cat(test_logits).argmax(1).numpy()

swa, cwa, pha = compute_metrics(test_seqs, y_test, test_pred)
print(f"\nTest  SWA={swa:.4f}  CWA={cwa:.4f}  PHA={pha:.4f}")

# save predictions and gt
experiment_data["spr_bench"]["predictions"] = test_pred
experiment_data["spr_bench"]["ground_truth"] = y_test
experiment_data["spr_bench"]["test_metrics"] = {"SWA": swa, "CWA": cwa, "PHA": pha}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ------------------------------------------------------------
# quick loss curve
plt.figure()
plt.plot(
    experiment_data["spr_bench"]["epochs"],
    experiment_data["spr_bench"]["losses"]["train"],
    label="train",
)
plt.plot(
    experiment_data["spr_bench"]["epochs"],
    experiment_data["spr_bench"]["losses"]["dev"],
    label="dev",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

print("All done; artefacts written to ./working")
