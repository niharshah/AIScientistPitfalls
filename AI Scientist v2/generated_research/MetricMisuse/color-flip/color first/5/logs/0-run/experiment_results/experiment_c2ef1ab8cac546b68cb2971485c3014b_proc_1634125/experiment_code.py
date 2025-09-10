import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ---------- paths & bookkeeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "weight_decay": {
        "SPR_BENCH": {
            "runs": {},  # each key will be the decay value as string
            "best_decay": None,
        }
    }
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)


# ---------- helper metrics ----------
def count_color_variety(seq: str):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str):
    return len({tok[0] for tok in seq.split() if tok})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- data ----------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("Dataset not found; generating synthetic sample")

    def synth_split(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        seqs, labels = [], []
        for _ in range(n):
            seq = " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            seqs.append(seq)
            labels.append(np.random.choice(["A", "B", "C"]))
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict(
        {
            sp: load_dataset(
                "json", data_files={"train": synth_split(n)}, split="train"
            )
            for sp, n in zip(["train", "dev", "test"], [200, 50, 50])
        }
    )

# ---------- vectorise ----------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])


def vec(lst: List[str]):
    return vectorizer.transform(lst).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vec,
    [dsets["train"]["sequence"], dsets["dev"]["sequence"], dsets["test"]["sequence"]],
)
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], np.int64)
num_classes = len(labels)
vocab_size = len(vectorizer.vocabulary_)
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=64,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=64
)


# ---------- model def ----------
class MLP(nn.Module):
    def __init__(self, inp, cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 256), nn.ReLU(), nn.Linear(256, cls))

    def forward(self, x):
        return self.net(x)


# ---------- training loop ----------
def train_run(weight_decay: float, epochs: int = 5):
    model = MLP(vocab_size, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    train_losses, val_losses, metrics_val = [], [], []
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()
            run_loss += loss.item() * xb.size(0)
        train_losses.append(run_loss / len(train_loader.dataset))
        # val
        model.eval()
        v_loss, preds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                v_loss.append(loss.item() * xb.size(0))
                preds.extend(logits.argmax(1).cpu().numpy())
        v_loss = sum(v_loss) / len(val_loader.dataset)
        val_losses.append(v_loss)
        tgts = y_val
        seqs = dsets["dev"]["sequence"]
        acc = (np.array(preds) == tgts).mean()
        cwa = color_weighted_accuracy(seqs, tgts, preds)
        swa = shape_weighted_accuracy(seqs, tgts, preds)
        comp = complexity_weighted_accuracy(seqs, tgts, preds)
        metrics_val.append(
            {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )
        print(
            f"wd={weight_decay:.0e}  epoch={ep}  train_loss={train_losses[-1]:.4f}  val_acc={acc:.3f}"
        )
    # test evaluation
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        test_preds = test_logits.argmax(1).cpu().numpy()
    seqs_test = dsets["test"]["sequence"]
    test_metrics = {
        "acc": (test_preds == y_test).mean(),
        "cwa": color_weighted_accuracy(seqs_test, y_test, test_preds),
        "swa": shape_weighted_accuracy(seqs_test, y_test, test_preds),
        "compwa": complexity_weighted_accuracy(seqs_test, y_test, test_preds),
    }
    return train_losses, val_losses, metrics_val, test_preds, test_metrics


# ---------- sweep ----------
decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
best_acc = -1
best_decay = None
for wd in decays:
    tr_losses, val_losses, metrics_val, test_preds, test_metrics = train_run(wd)
    key = str(wd)
    experiment_data["weight_decay"]["SPR_BENCH"]["runs"][key] = {
        "losses": {"train": tr_losses, "val": val_losses},
        "metrics": {"val": metrics_val, "test": test_metrics},
        "predictions": test_preds,
        "ground_truth": y_test.tolist(),
    }
    last_val_acc = metrics_val[-1]["acc"]
    if last_val_acc > best_acc:
        best_acc = last_val_acc
        best_decay = wd

experiment_data["weight_decay"]["SPR_BENCH"]["best_decay"] = best_decay
print(f"Best weight_decay: {best_decay} with validation acc {best_acc:.3f}")

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
