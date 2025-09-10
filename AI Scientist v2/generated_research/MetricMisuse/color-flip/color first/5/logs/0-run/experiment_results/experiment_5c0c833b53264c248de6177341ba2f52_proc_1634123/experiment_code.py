import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ------------------ bookkeeping ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "epochs_tuning": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "sequences": [],
            "best_epoch": None,
        }
    }
}
ed = experiment_data["epochs_tuning"]["SPR_BENCH"]  # shortcut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------ data -------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    # synthetic fallback
    print("SPR_BENCH not found – generating synthetic data.")
    shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]

    def synth_split(n):
        seqs, labels = [], []
        for _ in range(n):
            seqs.append(
                " ".join(
                    np.random.choice(
                        [s + c for s in shapes for c in colors],
                        size=np.random.randint(3, 8),
                    )
                )
            )
            labels.append(np.random.choice(["A", "B", "C"]))
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth_split(n)}, split="train"
        )


# ------------------ helpers ----------------------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def _wacc(seqs, y_true, y_pred, wfn):
    w = [wfn(s) for s in seqs]
    good = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    return _wacc(seqs, y_true, y_pred, count_color_variety)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    return _wacc(seqs, y_true, y_pred, count_shape_variety)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    return _wacc(
        seqs, y_true, y_pred, lambda s: count_color_variety(s) * count_shape_variety(s)
    )


# ------------- vectorise / encode ---------------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])


def vec(x: List[str]) -> np.ndarray:
    return vectorizer.transform(x).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vec,
    [dsets["train"]["sequence"], dsets["dev"]["sequence"], dsets["test"]["sequence"]],
)
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], np.int64)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=64,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=64
)


# ------------------ model ------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(len(vectorizer.vocabulary_), len(labels)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------ training w/ early stop -------
max_epochs, patience = 30, 5
best_val, best_epoch, no_improve = float("inf"), 0, 0
best_state = None

for epoch in range(1, max_epochs + 1):
    # --- train ---
    model.train()
    tot_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * xb.size(0)
    train_loss = tot_loss / len(train_loader.dataset)
    ed["losses"]["train"].append(train_loss)

    # --- val ---
    model.eval()
    v_loss, preds, tgts = 0.0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            v_loss += criterion(logits, yb).item() * xb.size(0)
            preds.extend(logits.argmax(1).cpu().numpy())
            tgts.extend(yb.cpu().numpy())
    v_loss /= len(val_loader.dataset)
    ed["losses"]["val"].append(v_loss)

    seqs_val = dsets["dev"]["sequence"]
    acc = (np.array(preds) == np.array(tgts)).mean()
    metrics_val = {
        "epoch": epoch,
        "acc": acc,
        "cwa": color_weighted_accuracy(seqs_val, tgts, preds),
        "swa": shape_weighted_accuracy(seqs_val, tgts, preds),
        "compwa": complexity_weighted_accuracy(seqs_val, tgts, preds),
    }
    ed["metrics"]["val"].append(metrics_val)
    ed["metrics"]["train"].append({"epoch": epoch, "loss": train_loss})

    print(
        f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  "
        f"val_loss={v_loss:.4f}  ACC={acc:.3f}"
    )

    # --- early stopping ---
    if v_loss + 1e-4 < best_val:
        best_val, best_epoch, no_improve = v_loss, epoch, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break

ed["best_epoch"] = best_epoch
# restore best model
model.load_state_dict(best_state)

# ------------------ test -------------------------
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(X_test).to(device))
    preds_test = logits.argmax(1).cpu().numpy()
test_acc = (preds_test == y_test).mean()
test_metrics = {
    "acc": test_acc,
    "cwa": color_weighted_accuracy(dsets["test"]["sequence"], y_test, preds_test),
    "swa": shape_weighted_accuracy(dsets["test"]["sequence"], y_test, preds_test),
    "compwa": complexity_weighted_accuracy(
        dsets["test"]["sequence"], y_test, preds_test
    ),
}
print(
    "\nTest:  ACC={acc:.3f}  CWA={cwa:.3f}  SWA={swa:.3f}  CompWA={compwa:.3f}".format(
        **test_metrics
    )
)

ed["predictions"] = preds_test
ed["ground_truth"] = y_test
ed["sequences"] = dsets["test"]["sequence"]
ed["metrics"]["test"] = test_metrics

# ------------------ save -------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to working/experiment_data.npy")
