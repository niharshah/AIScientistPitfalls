# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List, Tuple

# ---------- I/O ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "ngram_range_tuning": {
        "SPR_BENCH": {
            "runs": [],  # one entry per n-gram configuration
            "metrics": {"train": [], "val": []},  # of the best model
            "losses": {"train": [], "val": []},
            "best_ngram": None,
            "predictions": [],
            "ground_truth": [],
            "sequences": [],
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- Helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- Data ----------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:  # synthetic fallback
    print("SPR_BENCH not found, creating synthetic data.")

    def synth_split(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        seqs = [
            " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            for _ in range(n)
        ]
        labels = np.random.choice(["A", "B", "C"], size=n).tolist()
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth_split(n)}, split="train"
        )

# ---------- Labels ----------
labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], dtype=np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], dtype=np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], dtype=np.int64)
num_classes = len(labels)


# ---------- Model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, n_cls)
        )

    def forward(self, x):
        return self.net(x)


# ---------- Training routine ----------
def train_one_setting(ngram_range: Tuple[int, int]):
    # Vectoriser
    vect = CountVectorizer(token_pattern=r"[^ ]+", ngram_range=ngram_range)
    vect.fit(dsets["train"]["sequence"])

    def vec(seqs: List[str]):
        return vect.transform(seqs).toarray().astype(np.float32)

    X_tr, X_val, X_te = map(
        vec,
        [
            dsets["train"]["sequence"],
            dsets["dev"]["sequence"],
            dsets["test"]["sequence"],
        ],
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=64
    )
    model = MLP(X_tr.shape[1], num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    run_data = {
        "ngram": ngram_range,
        "losses": {"train": [], "val": []},
        "metrics": {"val": []},
    }

    for epoch in range(1, 6):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)
        # val
        model.eval()
        v_loss, preds = 0.0, []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = crit(out, yb)
                v_loss += loss.item() * xb.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
        v_loss /= len(val_loader.dataset)
        acc = (np.array(preds) == y_val).mean()
        cwa = color_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
        swa = shape_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
        comp = complexity_weighted_accuracy(dsets["dev"]["sequence"], y_val, preds)
        # log
        run_data["losses"]["train"].append(tr_loss)
        run_data["losses"]["val"].append(v_loss)
        run_data["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )
        print(
            f"ngram{ngram_range} epoch{epoch}: "
            f"train_loss={tr_loss:.4f} val_loss={v_loss:.4f} ACC={acc:.3f}"
        )
    return run_data, model, vect


# ---------- Hyper-parameter loop ----------
ngram_options = [(1, 1), (1, 2), (1, 3)]
best_val_acc, best_idx = -1, -1
models, vectors = [], []

for idx, ngr in enumerate(ngram_options):
    run, mdl, vec = train_one_setting(ngr)
    experiment_data["ngram_range_tuning"]["SPR_BENCH"]["runs"].append(run)
    models.append(mdl)
    vectors.append(vec)
    last_acc = run["metrics"]["val"][-1]["acc"]
    if last_acc > best_val_acc:
        best_val_acc, best_idx = last_acc, idx

# ---------- Test with best model ----------
best_model, best_vectorizer, best_ngram = (
    models[best_idx],
    vectors[best_idx],
    ngram_options[best_idx],
)
experiment_data["ngram_range_tuning"]["SPR_BENCH"]["best_ngram"] = best_ngram
best_model.eval()
X_test_best = (
    best_vectorizer.transform(dsets["test"]["sequence"]).toarray().astype(np.float32)
)
with torch.no_grad():
    preds = best_model(torch.from_numpy(X_test_best).to(device)).argmax(1).cpu().numpy()

seq_test = dsets["test"]["sequence"]
test_acc = (preds == y_test).mean()
test_cwa = color_weighted_accuracy(seq_test, y_test, preds)
test_swa = shape_weighted_accuracy(seq_test, y_test, preds)
test_comp = complexity_weighted_accuracy(seq_test, y_test, preds)

experiment_data["ngram_range_tuning"]["SPR_BENCH"]["predictions"] = preds
experiment_data["ngram_range_tuning"]["SPR_BENCH"]["ground_truth"] = y_test
experiment_data["ngram_range_tuning"]["SPR_BENCH"]["sequences"] = seq_test
experiment_data["ngram_range_tuning"]["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "cwa": test_cwa,
    "swa": test_swa,
    "compwa": test_comp,
}

print(
    f"\nBest n-gram {best_ngram} — Test ACC={test_acc:.3f} "
    f"CWA={test_cwa:.3f} SWA={test_swa:.3f} CompWA={test_comp:.3f}"
)

# ---------- Save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
