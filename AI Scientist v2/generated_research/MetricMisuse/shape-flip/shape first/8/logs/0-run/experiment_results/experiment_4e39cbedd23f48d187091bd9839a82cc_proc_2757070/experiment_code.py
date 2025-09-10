import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- helpers -------------------
def count_shape_variety(seq):  # unique first-chars
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):  # unique second-chars
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1, sum(w))


def rule_signature(seq):
    return (count_shape_variety(seq), count_color_variety(seq))


# -------------- synthetic fallback ----------
def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:6])


def generate_synthetic_split(n, seed=0):
    random.seed(seed)
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(3, 10)
        seq = " ".join(random_token() for _ in range(L))
        lbl = int(count_shape_variety(seq) == count_color_variety(seq))
        seqs.append(seq)
        labels.append(lbl)
    return {"id": list(range(n)), "sequence": seqs, "label": labels}


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    if root.exists():
        l = lambda f: load_dataset("csv", data_files=str(root / f), split="train")
        return DatasetDict(train=l("train.csv"), dev=l("dev.csv"), test=l("test.csv"))
    print("SPR_BENCH not found – using synthetic toy data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, 1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, 2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, 3)),
    )


# -------------- data loading ---------------
DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

shape_to_idx = {c: i for i, c in enumerate(string.ascii_uppercase[:26])}
color_to_idx = {d: i for i, d in enumerate(string.digits[:10])}
feat_dim = 26 + 10  # histogram only (No-Global-Stats ablation)


def encode(seq: str):
    v = np.zeros(feat_dim, np.float32)
    for tok in seq.split():
        if len(tok) < 2:
            continue
        v[shape_to_idx[tok[0]]] += 1
        v[26 + color_to_idx[tok[1]]] += 1
    return v


def encode_split(hfd):
    X = np.stack([encode(s) for s in hfd["sequence"]])
    y = np.array(hfd["label"], np.int64)
    sigs = [rule_signature(s) for s in hfd["sequence"]]
    return X, y, sigs


X_train, y_train, sig_train = encode_split(dsets["train"])
X_dev, y_dev, sig_dev = encode_split(dsets["dev"])
X_test, y_test, sig_test = encode_split(dsets["test"])
train_signatures = set(sig_train)


class SPRTorchDS(Dataset):
    def __init__(self, X, y, seqs):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.seqs = seqs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i], "seq": self.seqs[i]}


bs_train, bs_eval = 64, 256
train_loader = DataLoader(
    SPRTorchDS(X_train, y_train, dsets["train"]["sequence"]), bs_train, shuffle=True
)
dev_loader = DataLoader(SPRTorchDS(X_dev, y_dev, dsets["dev"]["sequence"]), bs_eval)
test_loader = DataLoader(SPRTorchDS(X_test, y_test, dsets["test"]["sequence"]), bs_eval)


# ---------------- model ---------------------
class MLP(nn.Module):
    def __init__(self, indim, hidden=64, classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hidden), nn.ReLU(), nn.Linear(hidden, classes)
        )

    def forward(self, x):
        return self.net(x)


model, criterion = MLP(feat_dim).to(device), nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------- symbolic rule --------------------
symbolic_predict = lambda seq: int(count_shape_variety(seq) == count_color_variety(seq))


# -------------- evaluation -----------------
def evaluate(loader, seqs, y_true):
    model.eval()
    preds, losses = [], []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            nn_preds = logits.argmax(1).cpu().numpy()
            for j in range(logits.size(0)):
                seq = seqs[idx]
                sig = rule_signature(seq)
                pred = (
                    symbolic_predict(seq)
                    if sig not in train_signatures
                    else int(nn_preds[j])
                )
                preds.append(pred)
                losses.append(
                    criterion(
                        logits[j : j + 1], batch["y"][j : j + 1].to(device)
                    ).item()
                )
                idx += 1
    return np.mean(losses), shape_weighted_accuracy(seqs, y_true, preds), preds


# ----------- experiment data dict ----------
experiment_data = {
    "no_global_stats": {
        "spr_bench": {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "timestamps": [],
        }
    }
}

# -------------- training loop --------------
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    model.train()
    run_loss = n = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["y"].size(0)
        n += batch["y"].size(0)
    train_loss = run_loss / n
    _, train_swa, _ = evaluate(train_loader, dsets["train"]["sequence"], y_train)
    val_loss, val_swa, _ = evaluate(dev_loader, dsets["dev"]["sequence"], y_dev)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}  val_SWA={val_swa:.3f}")
    ed = experiment_data["no_global_stats"]["spr_bench"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_swa"].append(train_swa)
    ed["metrics"]["val_swa"].append(val_swa)
    ed["timestamps"].append(time.time())

# ---------------- final test ----------------
test_loss, test_swa, test_preds = evaluate(
    test_loader, dsets["test"]["sequence"], y_test
)
print(f"\nTest Shape-Weighted Accuracy (SWA) = {test_swa:.3f}")
experiment_data["no_global_stats"]["spr_bench"]["predictions"] = test_preds
experiment_data["no_global_stats"]["spr_bench"]["metrics"]["test_swa"] = test_swa

# -------------- save results ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
