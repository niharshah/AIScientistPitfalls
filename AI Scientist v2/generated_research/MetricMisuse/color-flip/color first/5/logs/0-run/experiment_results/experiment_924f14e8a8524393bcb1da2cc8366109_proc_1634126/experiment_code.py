import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ---------- directories & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment dict ----------
experiment_data = {
    "batch_size": {"SPR_BENCH": {}}  # will be filled with one sub-dict per batch size
}


# ---------- metric helpers ----------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def _w_acc(seqs, y_t, y_p, weight_fn):
    w = [weight_fn(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    return _w_acc(seqs, y_t, y_p, count_color_variety)


def shape_weighted_accuracy(seqs, y_t, y_p):
    return _w_acc(seqs, y_t, y_p, count_shape_variety)


def complexity_weighted_accuracy(seqs, y_t, y_p):
    return _w_acc(
        seqs, y_t, y_p, lambda s: count_color_variety(s) * count_shape_variety(s)
    )


# ---------- Load data (real or synthetic) ----------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():

    def _load(csv_name):  # helper for csv splits
        return load_dataset(
            "csv",
            data_files=str(pathlib.Path(DATA_ENV) / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dsets = DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )
else:  # synthetic fallback
    print("SPR_BENCH not found. Creating synthetic data for demo.")
    shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]

    def synth_split(n):
        seqs, labs = [], []
        for _ in range(n):
            seq = " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            labs.append(np.random.choice(["A", "B", "C"]))
            seqs.append(seq)
        return {"sequence": seqs, "label": labs}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth_split(n)}, split="train"
        )

# ---------- Vectorisation ----------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])
vocab_size = len(vectorizer.vocabulary_)
print(f"Vocabulary size: {vocab_size}")


def vec(seqs: List[str]) -> np.ndarray:
    return vectorizer.transform(seqs).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vec,
    (dsets["train"]["sequence"], dsets["dev"]["sequence"], dsets["test"]["sequence"]),
)

# ---------- Label encoding ----------
labels = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
num_classes = len(labels)
y_train = np.array([lab2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([lab2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([lab2id[l] for l in dsets["test"]["label"]], np.int64)
print(f"Number of classes: {num_classes}")


# ---------- Model definition ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Hyperparameter sweep ----------
batch_sizes = [32, 64, 128, 256]
epochs = 5

for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "sequences": dsets["test"]["sequence"],
    }
    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = exp_rec

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=bs,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=bs
    )

    model = MLP(vocab_size, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        # --- train ---
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)
        exp_rec["losses"]["train"].append(tr_loss)

        # --- validate ---
        model.eval()
        val_loss, preds, tgts = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += crit(logits, yb).item() * xb.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                tgts.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        exp_rec["losses"]["val"].append(val_loss)

        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        cwa = color_weighted_accuracy(seqs_val, tgts, preds)
        swa = shape_weighted_accuracy(seqs_val, tgts, preds)
        comp = complexity_weighted_accuracy(seqs_val, tgts, preds)
        exp_rec["metrics"]["val"].append(
            {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )
        print(
            f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )

    # --- final test evaluation ---
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        test_preds = test_logits.argmax(1).cpu().numpy()
    exp_rec["predictions"] = test_preds.tolist()
    t_acc = (test_preds == y_test).mean()
    t_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    t_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    t_comp = complexity_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    exp_rec["metrics"]["test"] = {
        "acc": t_acc,
        "cwa": t_cwa,
        "swa": t_swa,
        "compwa": t_comp,
    }
    print(f"Test — ACC={t_acc:.3f} CWA={t_cwa:.3f} SWA={t_swa:.3f} CompWA={t_comp:.3f}")

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nSaved results to {os.path.join(working_dir, 'experiment_data.npy')}")
