# joint_token_only_ablation.py
import os, pathlib, random, string, warnings, sys, json, time
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- paths / device ---------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ---------- try import helpers -----------------------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception as e:
    warnings.warn("Could not import SPR helpers, falling back to simple versions")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError("SPR_BENCH not found")

    def _variety(sequence, idx):
        return len({tok[idx] for tok in sequence.strip().split() if len(tok) > idx})

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_variety(s, 0) for s in seqs]
        correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(correct) / (sum(w) + 1e-9)

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [_variety(s, 1) for s in seqs]
        correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(correct) / (sum(w) + 1e-9)


# ---------- synthetic fallback dataset ---------------------------------------
def make_synth(n):
    shapes = list(string.ascii_uppercase[:6])
    cols = list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n):
        length = random.randint(4, 9)
        toks = [random.choice(shapes) + random.choice(cols) for _ in range(length)]
        seqs.append(" ".join(toks))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ---------- load data ---------------------------------------------------------
root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    d = load_spr_bench(root)
    print("Loaded real SPR_BENCH.")
    train_seqs, train_labels = d["train"]["sequence"], d["train"]["label"]
    dev_seqs, dev_labels = d["dev"]["sequence"], d["dev"]["label"]
    test_seqs, test_labels = d["test"]["sequence"], d["test"]["label"]
except Exception as e:
    warnings.warn(f"{e}\nUsing synthetic data.")
    train, dev, test = make_synth(512), make_synth(128), make_synth(256)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

# ---------- JOINT-TOKEN vocabulary & featuriser -------------------------------
joint_vocab = sorted({tok for seq in train_seqs for tok in seq.split() if tok})
tok2idx = {t: i for i, t in enumerate(joint_vocab)}
feat_dim = len(tok2idx)
print(f"Joint-token vocab size / feature dim = {feat_dim}")


def seq_to_feature(seq: str) -> np.ndarray:
    vec = np.zeros(feat_dim, dtype=np.float32)
    for tok in seq.split():
        if tok in tok2idx:
            vec[tok2idx[tok]] += 1.0
    return vec


def encode_dataset(seqs, labels):
    X = np.stack([seq_to_feature(s) for s in seqs])
    y = np.asarray(labels, dtype=np.int64)
    return X, y


X_train, y_train = encode_dataset(train_seqs, train_labels)
X_dev, y_dev = encode_dataset(dev_seqs, dev_labels)
X_test, y_test = encode_dataset(test_seqs, test_labels)
n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"Detected {n_classes} classes")

# ---------- dataloaders -------------------------------------------------------
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


# ---------- simple MLP --------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, n_out))

    def forward(self, x):
        return self.net(x)


def compute_metrics(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    pha = 2 * swa * cwa / (swa + cwa + 1e-9)
    return swa, cwa, pha


# ---------- experiment data dict ---------------------------------------------
experiment_data = {
    "joint_token_only": {
        "spr_bench": {
            "metrics": {"train_PHA": [], "dev_PHA": []},
            "losses": {"train": [], "dev": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ---------- training loop with early stopping --------------------------------
model = MLP(feat_dim, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
max_epochs, patience = 50, 7
best_dev_pha, wait, best_state = -1.0, 0, None

for epoch in range(1, max_epochs + 1):
    # ---- train
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optim.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_loader.dataset)

    # ---- val
    model.eval()
    running = 0.0
    dev_logits, dev_ys = [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            lg = model(xb)
            running += criterion(lg, yb).item() * xb.size(0)
            dev_logits.append(lg.cpu())
            dev_ys.append(yb.cpu())
    dev_loss = running / len(dev_loader.dataset)
    dev_pred = torch.cat(dev_logits).argmax(1).numpy()
    dev_gt = torch.cat(dev_ys).numpy()

    # ---- metrics
    _, _, train_pha = compute_metrics(
        train_seqs,
        y_train,
        model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy(),
    )
    _, _, dev_pha = compute_metrics(dev_seqs, y_dev, dev_pred)

    # ---- log
    log = experiment_data["joint_token_only"]["spr_bench"]
    log["epochs"].append(epoch)
    log["losses"]["train"].append(train_loss)
    log["losses"]["dev"].append(dev_loss)
    log["metrics"]["train_PHA"].append(train_pha)
    log["metrics"]["dev_PHA"].append(dev_pha)

    print(
        f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | dev_loss {dev_loss:.4f} | dev_PHA {dev_pha:.4f}"
    )

    if dev_pha > best_dev_pha + 1e-5:
        best_dev_pha, wait = dev_pha, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# ---------- restore best model -----------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

# ---------- test evaluation ---------------------------------------------------
model.eval()
test_logits = []
with torch.no_grad():
    for xb, _ in test_loader:
        test_logits.append(model(xb.to(device)).cpu())
test_pred = torch.cat(test_logits).argmax(1).numpy()
swa, cwa, pha = compute_metrics(test_seqs, y_test, test_pred)
print(f"\nTEST | SWA {swa:.4f} | CWA {cwa:.4f} | PHA {pha:.4f}")

log["predictions"] = test_pred
log["ground_truth"] = y_test
log["test_metrics"] = {"SWA": swa, "CWA": cwa, "PHA": pha}

# ---------- save artefacts ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure()
plt.plot(log["epochs"], log["losses"]["train"], label="train")
plt.plot(log["epochs"], log["losses"]["dev"], label="dev")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve (Joint token only)")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

print("All done. Artefacts saved to ./working")
