import os, pathlib, random, string, warnings, time
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------- work dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# -------------------------------------------------- helper import / fall-backs
try:
    from SPR import (
        load_spr_bench,
        count_shape_variety,
        count_color_variety,
        shape_weighted_accuracy,
    )
except Exception as e:
    warnings.warn(f"Fallback helpers because SPR import failed: {e}")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError

    def count_shape_variety(seq: str) -> int:
        return len(set(tok[0] for tok in seq.split() if tok))

    def count_color_variety(seq: str) -> int:
        return len(set(tok[1] for tok in seq.split() if len(tok) > 1))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [count_shape_variety(s) for s in seqs]
        correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(correct) / (sum(w) + 1e-9)


# -------------------------------------------------- if benchmark unavailable build tiny synthetic set
def make_synth(n):
    shapes, cols = list(string.ascii_uppercase[:6]), list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n):
        tokens = [
            random.choice(shapes) + random.choice(cols)
            for _ in range(random.randint(4, 10))
        ]
        seqs.append(" ".join(tokens))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root)
    train_seqs, train_y = dsets["train"]["sequence"], dsets["train"]["label"]
    dev_seqs, dev_y = dsets["dev"]["sequence"], dsets["dev"]["label"]
    test_seqs, test_y = dsets["test"]["sequence"], dsets["test"]["label"]
    print("Loaded real SPR_BENCH")
except Exception as e:
    warnings.warn(f"{e}\nGenerating synthetic toy data instead")
    tr, dv, te = make_synth(800), make_synth(200), make_synth(400)
    train_seqs, train_y = tr["sequence"], tr["label"]
    dev_seqs, dev_y = dv["sequence"], dv["label"]
    test_seqs, test_y = te["sequence"], te["label"]

# -------------------------------------------------- vocab build
shapes = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
colors = sorted({tok[1] for seq in train_seqs for tok in seq.split() if len(tok) > 1})
shape2i = {s: i for i, s in enumerate(shapes)}
color2i = {c: i for i, c in enumerate(colors)}
pair_dim = len(shapes) * len(colors)
feat_dim = pair_dim + len(shapes) + len(colors) + 2  # +2 for variety cues


def seq_to_feat(seq: str):
    pair = np.zeros(pair_dim, np.float32)
    sh = np.zeros(len(shapes), np.float32)
    co = np.zeros(len(colors), np.float32)
    for tok in seq.split():
        if not tok:
            continue
        s, c = tok[0], tok[1] if len(tok) > 1 else None
        sid = shape2i.get(s, None)
        cid = color2i.get(c, None) if c else None
        if sid is not None:
            sh[sid] += 1
        if cid is not None:
            co[cid] += 1
        if sid is not None and cid is not None:
            pair[sid * len(colors) + cid] += 1
    svar, cvar = count_shape_variety(seq), count_color_variety(seq)
    return np.concatenate([pair, sh, co, np.asarray([svar, cvar], np.float32)])


def encode_dataset(seqs, labels):
    X = np.stack([seq_to_feat(s) for s in seqs])
    y = np.asarray(labels, dtype=np.int64)
    w = np.asarray(
        [count_shape_variety(s) for s in seqs], dtype=np.float32
    )  # training weight
    return X, y, w


X_tr, y_tr, w_tr = encode_dataset(train_seqs, train_y)
X_dv, y_dv, _ = encode_dataset(dev_seqs, dev_y)
X_te, y_te, _ = encode_dataset(test_seqs, test_y)

n_classes = int(max(y_tr.max(), y_dv.max(), y_te.max())) + 1
print("Classes:", n_classes, "Feature dim:", feat_dim)

# -------------------------------------------------- dataloaders
bs = 128
train_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(w_tr)
    ),
    batch_size=bs,
    shuffle=True,
)
dev_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_dv), torch.from_numpy(y_dv)), batch_size=bs
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=bs
)


# -------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, d_in, nc):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, nc),
        )

    def forward(self, x):
        return self.seq(x)


model = MLP(feat_dim, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(reduction="none")  # we'll apply weights

# -------------------------------------------------- storage dict
experiment_data = {
    "SPR_SWA": {
        "epochs": [],
        "losses": {"train": [], "dev": []},
        "metrics": {"dev_SWA": []},
        "predictions": [],
        "ground_truth": y_te,
    }
}

# -------------------------------------------------- training
best_swa, patience, waited = -1.0, 8, 0
max_epochs = 60
for epoch in range(1, max_epochs + 1):
    model.train()
    running = 0.0
    for xb, yb, wb in train_loader:
        xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss = (loss * wb).mean()  # weight by shape-variety
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_loader.dataset)

    # ----- dev
    model.eval()
    total, dev_logits = [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            lg = model(xb)
            dev_logits.append(lg.cpu())
            total.append(nn.functional.cross_entropy(lg, yb, reduction="sum").item())
    dev_loss = sum(total) / len(dev_loader.dataset)
    dev_pred = torch.cat(dev_logits).argmax(1).numpy()
    dev_swa = shape_weighted_accuracy(dev_seqs, y_dv, dev_pred)

    experiment_data["SPR_SWA"]["epochs"].append(epoch)
    experiment_data["SPR_SWA"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_SWA"]["losses"]["dev"].append(dev_loss)
    experiment_data["SPR_SWA"]["metrics"]["dev_SWA"].append(dev_swa)
    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_SWA={dev_swa:.4f}"
    )

    if dev_swa > best_swa + 1e-5:
        best_swa = dev_swa
        waited = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        waited += 1
        if waited >= patience:
            print("Early stopping")
            break

# -------------------------------------------------- test evaluation
model.load_state_dict(best_state)
model.eval()
test_logits = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        test_logits.append(model(xb).cpu())
test_pred = torch.cat(test_logits).argmax(1).numpy()
test_swa = shape_weighted_accuracy(test_seqs, y_te, test_pred)
print(f"\nTEST Shape-Weighted Accuracy (SWA) = {test_swa:.4f}")

experiment_data["SPR_SWA"]["predictions"] = test_pred
experiment_data["SPR_SWA"]["test_SWA"] = test_swa
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Artifacts saved to", working_dir)
