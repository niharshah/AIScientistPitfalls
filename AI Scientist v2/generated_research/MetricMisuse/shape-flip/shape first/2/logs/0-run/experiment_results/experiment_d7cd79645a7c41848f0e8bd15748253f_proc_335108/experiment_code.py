# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Remove-Shape-Features (Color-Only Model)
import os, pathlib, random, string, warnings, sys
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------- paths / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------- helpers / metrics
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception as e:
    warnings.warn("Could not import SPR helpers, falling back to local impl.")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError("SPR_BENCH not found")

    def _count_shape(seq):  # still needed for metric
        return len(set(tok[0] for tok in seq.strip().split() if tok))

    def _count_color(seq):
        return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_shape(s) for s in seqs]
        c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(c) / (sum(w) + 1e-9)

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_color(s) for s in seqs]
        c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(c) / (sum(w) + 1e-9)


# ------------------------------- synthetic dataset helper
def make_synthetic_dataset(n_rows):
    shapes = list(string.ascii_uppercase[:6])
    cols = list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n_rows):
        length = random.randint(4, 9)
        tokens = [random.choice(shapes) + random.choice(cols) for _ in range(length)]
        seqs.append(" ".join(tokens))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ------------------------------- load data
root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root)
    print("Loaded real SPR_BENCH")
    train_seqs, train_labels = dsets["train"]["sequence"], dsets["train"]["label"]
    dev_seqs, dev_labels = dsets["dev"]["sequence"], dsets["dev"]["label"]
    test_seqs, test_labels = dsets["test"]["sequence"], dsets["test"]["label"]
except Exception as e:
    warnings.warn(f"{e}\nUsing synthetic data.")
    train = make_synthetic_dataset(512)
    dev = make_synthetic_dataset(128)
    test = make_synthetic_dataset(256)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

# ------------------------------- vocab (colors only)
color_vocab = sorted(
    {tok[1] for seq in train_seqs for tok in seq.split() if len(tok) > 1}
)
color2idx = {c: i for i, c in enumerate(color_vocab)}
feat_dim = len(color2idx)
print(f"Color-only feature dim = {feat_dim}")


# ------------------------------- featuriser
def seq_to_feature(seq: str) -> np.ndarray:
    col_vec = np.zeros(feat_dim, dtype=np.float32)
    for tok in seq.split():
        if len(tok) > 1:
            idx = color2idx.get(tok[1])
            if idx is not None:
                col_vec[idx] += 1.0
    return col_vec


def encode_dataset(seqs, labels):
    X = np.stack([seq_to_feature(s) for s in seqs])
    y = np.asarray(labels, dtype=np.int64)
    return X, y


X_train, y_train = encode_dataset(train_seqs, train_labels)
X_dev, y_dev = encode_dataset(dev_seqs, dev_labels)
X_test, y_test = encode_dataset(test_seqs, test_labels)
n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"Detected {n_classes} classes")

batch_size = 128
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=batch_size,
    shuffle=True,
)
dev_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev)),
    batch_size=batch_size,
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=batch_size,
)


# ------------------------------- model
class MLP(nn.Module):
    def __init__(self, in_dim, nc):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, nc))

    def forward(self, x):
        return self.net(x)


def compute_metrics(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    pha = 2 * swa * cwa / (swa + cwa + 1e-9)
    return swa, cwa, pha


# ------------------------------- experiment container
experiment_data = {
    "remove_shape_features": {
        "spr_bench": {
            "metrics": {"train_PHA": [], "dev_PHA": []},
            "losses": {"train": [], "dev": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ------------------------------- training
max_epochs, patience = 50, 7
model = MLP(feat_dim, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
best_dev_pha, wait, best_state = -1.0, 0, None

for epoch in range(1, max_epochs + 1):
    # -- train
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optim.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)

    # -- validate
    model.eval()
    dev_loss, logits_lst, ys_lst = 0.0, [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            dev_loss += criterion(logits, yb).item() * xb.size(0)
            logits_lst.append(logits.cpu())
            ys_lst.append(yb.cpu())
    dev_loss /= len(dev_loader.dataset)
    dev_pred = torch.cat(logits_lst).argmax(1).numpy()
    dev_gt = torch.cat(ys_lst).numpy()

    # -- metrics
    _, _, train_pha = compute_metrics(
        train_seqs,
        y_train,
        model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy(),
    )
    _, _, dev_pha = compute_metrics(dev_seqs, y_dev, dev_pred)

    # -- log
    log = experiment_data["remove_shape_features"]["spr_bench"]
    log["epochs"].append(epoch)
    log["losses"]["train"].append(tr_loss)
    log["losses"]["dev"].append(dev_loss)
    log["metrics"]["train_PHA"].append(train_pha)
    log["metrics"]["dev_PHA"].append(dev_pha)

    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} dev_loss={dev_loss:.4f} dev_PHA={dev_pha:.4f}"
    )

    # -- early stopping
    if dev_pha > best_dev_pha + 1e-5:
        best_dev_pha = dev_pha
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# ------------------------------- restore best
if best_state is not None:
    model.load_state_dict(best_state)

# ------------------------------- test
model.eval()
test_logits = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        test_logits.append(model(xb).cpu())
test_pred = torch.cat(test_logits).argmax(1).numpy()
swa, cwa, pha = compute_metrics(test_seqs, y_test, test_pred)
print(f"\nTest SWA={swa:.4f} CWA={cwa:.4f} PHA={pha:.4f}")

log = experiment_data["remove_shape_features"]["spr_bench"]
log["predictions"] = test_pred
log["ground_truth"] = y_test
log["test_metrics"] = {"SWA": swa, "CWA": cwa, "PHA": pha}

# ------------------------------- save artefacts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure()
plt.plot(log["epochs"], log["losses"]["train"], label="train")
plt.plot(log["epochs"], log["losses"]["dev"], label="dev")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Color-Only Loss Curve")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()
print("All done; artefacts saved to ./working")
