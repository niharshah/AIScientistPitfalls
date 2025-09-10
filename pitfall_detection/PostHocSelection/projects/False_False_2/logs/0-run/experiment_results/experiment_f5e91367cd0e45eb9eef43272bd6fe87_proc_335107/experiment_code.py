import os, pathlib, random, string, warnings, sys, time, json
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------- paths / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------- try import helpers
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception as e:
    warnings.warn("Could not import SPR helpers, using fall-backs")

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
root_path = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root_path)
    print("Loaded real SPR_BENCH.")
    train_seqs, train_labels = dsets["train"]["sequence"], dsets["train"]["label"]
    dev_seqs, dev_labels = dsets["dev"]["sequence"], dsets["dev"]["label"]
    test_seqs, test_labels = dsets["test"]["sequence"], dsets["test"]["label"]
except Exception as e:
    warnings.warn(f"{e}\nGenerating synthetic data instead.")
    train = make_synthetic_dataset(512)
    dev = make_synthetic_dataset(128)
    test = make_synthetic_dataset(256)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

# ------------------------------- vocab / featuriser (SHAPE ONLY)
shape_vocab = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
shape2idx = {s: i for i, s in enumerate(shape_vocab)}
feat_dim = len(shape2idx)
print(f"Feature dim (shape only) = {feat_dim}")


def seq_to_feature(seq: str) -> np.ndarray:
    sh = np.zeros(len(shape2idx), dtype=np.float32)
    for tok in seq.split():
        if not tok:
            continue
        sh[shape2idx.get(tok[0], 0)] += 1.0
    return sh


def encode_dataset(seqs, labels):
    X = np.stack([seq_to_feature(s) for s in seqs])
    y = np.asarray(labels, dtype=np.int64)
    return X, y


X_train, y_train = encode_dataset(train_seqs, train_labels)
X_dev, y_dev = encode_dataset(dev_seqs, dev_labels)
X_test, y_test = encode_dataset(test_seqs, test_labels)
n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"Detected {n_classes} classes")

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


# ------------------------------- model def
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


# ------------------------------- experiment data container
experiment_data = {
    "remove_color_features": {
        "spr_bench": {
            "metrics": {"train_PHA": [], "dev_PHA": []},
            "losses": {"train": [], "dev": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ------------------------------- training with early stopping
max_epochs = 50
patience = 7
model = MLP(feat_dim, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_dev_pha, wait = -1.0, 0
best_state = None

for epoch in range(1, max_epochs + 1):
    # ---- train
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_loader.dataset)

    # ---- validate
    model.eval()
    running = 0.0
    dev_logits, dev_ys = [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            running += criterion(logits, yb).item() * xb.size(0)
            dev_logits.append(logits.cpu())
            dev_ys.append(yb.cpu())
    dev_loss = running / len(dev_loader.dataset)
    dev_pred = torch.cat(dev_logits).argmax(1).numpy()
    dev_gt = torch.cat(dev_ys).numpy()

    # ---- PHA metrics
    _, _, train_pha = compute_metrics(
        train_seqs,
        y_train,
        model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy(),
    )
    _, _, dev_pha = compute_metrics(dev_seqs, y_dev, dev_pred)

    # ---- log
    ep_log = experiment_data["remove_color_features"]["spr_bench"]
    ep_log["epochs"].append(epoch)
    ep_log["losses"]["train"].append(train_loss)
    ep_log["losses"]["dev"].append(dev_loss)
    ep_log["metrics"]["train_PHA"].append(train_pha)
    ep_log["metrics"]["dev_PHA"].append(dev_pha)

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_PHA={dev_pha:.4f}"
    )

    # ---- early stopping on dev PHA
    if dev_pha > best_dev_pha + 1e-5:
        best_dev_pha = dev_pha
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ------------------------------- restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# ------------------------------- test evaluation
model.eval()
test_logits = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        test_logits.append(model(xb).cpu())
test_pred = torch.cat(test_logits).argmax(1).numpy()
swa, cwa, pha = compute_metrics(test_seqs, y_test, test_pred)
print(f"\nTest SWA={swa:.4f} CWA={cwa:.4f} PHA={pha:.4f}")

# save predictions & gt
ep_log = experiment_data["remove_color_features"]["spr_bench"]
ep_log["predictions"] = test_pred
ep_log["ground_truth"] = y_test
ep_log["test_metrics"] = {"SWA": swa, "CWA": cwa, "PHA": pha}

# ------------------------------- save artefacts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ------------------------------- plot loss curves
plt.figure()
plt.plot(ep_log["epochs"], ep_log["losses"]["train"], label="train")
plt.plot(ep_log["epochs"], ep_log["losses"]["dev"], label="dev")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (Remove-Color-Features)")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

print("All done; artefacts written to ./working")
