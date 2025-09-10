import os, pathlib, random, string, warnings, sys, time, json
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------------------------------------
# helper imports (fallbacks if SPR unavailable)
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy
except Exception as e:
    warnings.warn("Could not import SPR helpers, using fallbacks")

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
def make_synthetic_dataset(n_rows):
    shapes = list(string.ascii_uppercase[:6])
    cols = list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n_rows):
        length = random.randint(4, 9)
        toks = [random.choice(shapes) + random.choice(cols) for _ in range(length)]
        seqs.append(" ".join(toks))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ------------------------------------------------------------
# load data
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

# ------------------------------------------------------------
# build vocab
shape_vocab = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
color_vocab = sorted(
    {tok[1] for seq in train_seqs for tok in seq.split() if len(tok) > 1}
)
shape2idx = {s: i for i, s in enumerate(shape_vocab)}
color2idx = {c: i for i, c in enumerate(color_vocab)}
feat_dim = len(shape2idx) + len(color2idx)
print(f"Feature dim = {feat_dim}")


def seq_to_feature(seq: str) -> np.ndarray:
    sh = np.zeros(len(shape2idx), dtype=np.float32)
    co = np.zeros(len(color2idx), dtype=np.float32)
    for tok in seq.split():
        if not tok:
            continue
        sh[shape2idx.get(tok[0], 0)] += 1.0
        if len(tok) > 1:
            co[color2idx.get(tok[1], 0)] += 1.0
    return np.concatenate([sh, co])


def encode(seqs, labels):
    X = np.stack([seq_to_feature(s) for s in seqs])
    y = np.array(labels, dtype=np.int64)
    return X, y


X_train, y_train = encode(train_seqs, train_labels)
X_dev, y_dev = encode(dev_seqs, dev_labels)
X_test, y_test = encode(test_seqs, test_labels)
n_classes = int(max(y_train.max(), y_dev.max(), y_test.max())) + 1
print(f"Detected {n_classes} classes")


# ------------------------------------------------------------
def compute_metrics(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    pha = 2 * swa * cwa / (swa + cwa + 1e-9)
    return swa, cwa, pha


class MLP(nn.Module):
    def __init__(self, inp, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 128), nn.ReLU(), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
experiment_data = {"batch_size": {}}  # hyperparameter tuning container
batch_sizes = [32, 64, 128, 256]
epochs = 10

for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    # dataloaders
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

    model = MLP(feat_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log = {
        "metrics": {"train_PHA": [], "dev_PHA": []},
        "losses": {"train": [], "dev": []},
        "epochs": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # dev
        model.eval()
        dev_loss, dev_preds, dev_ys = 0.0, [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                dev_loss += criterion(logits, yb).item() * xb.size(0)
                dev_preds.append(logits.cpu())
                dev_ys.append(yb.cpu())
        dev_loss /= len(dev_loader.dataset)
        dev_preds = torch.cat(dev_preds).argmax(1).numpy()
        dev_ys = torch.cat(dev_ys).numpy()

        # PHA
        _, _, tr_pha = compute_metrics(
            train_seqs,
            y_train,
            model(torch.from_numpy(X_train).to(device)).argmax(1).cpu().numpy(),
        )
        _, _, dv_pha = compute_metrics(dev_seqs, y_dev, dev_preds)

        # logging
        log["metrics"]["train_PHA"].append(tr_pha)
        log["metrics"]["dev_PHA"].append(dv_pha)
        log["losses"]["train"].append(tr_loss)
        log["losses"]["dev"].append(dev_loss)
        log["epochs"].append(epoch)
        print(
            f"bs={bs}  epoch={epoch}  train_loss={tr_loss:.4f}  dev_loss={dev_loss:.4f}  dev_PHA={dv_pha:.4f}"
        )

    # final test
    model.eval()
    tst_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            tst_preds.append(model(xb.to(device)).cpu())
    tst_preds = torch.cat(tst_preds).argmax(1).numpy()
    swa, cwa, pha = compute_metrics(test_seqs, y_test, tst_preds)
    print(f"batch_size={bs}  Test  SWA={swa:.4f}  CWA={cwa:.4f}  PHA={pha:.4f}")

    log["predictions"] = tst_preds
    log["ground_truth"] = y_test
    log["test_metrics"] = {"SWA": swa, "CWA": cwa, "PHA": pha}
    experiment_data["batch_size"][str(bs)] = log

    # plot loss curve for this batch size
    plt.figure()
    plt.plot(log["epochs"], log["losses"]["train"], label="train")
    plt.plot(log["epochs"], log["losses"]["dev"], label="dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve (bs={bs})")
    plt.savefig(os.path.join(working_dir, f"loss_curve_bs{bs}.png"))
    plt.close()

# ------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll done; artefacts written to ./working")
