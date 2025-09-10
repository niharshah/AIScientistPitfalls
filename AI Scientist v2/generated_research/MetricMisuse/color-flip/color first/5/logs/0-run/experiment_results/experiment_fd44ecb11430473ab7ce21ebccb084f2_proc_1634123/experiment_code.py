import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ---------- Book-keeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "dropout_rate": {"SPR_BENCH": {}}  # hyper-parameter type  # dataset name
}


# ---------- Helpers & metrics ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_file):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def weighted_acc(seqs, y_true, y_pred, w_fn):
    w = [w_fn(s) for s in seqs]
    cor = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(s, y_t, y_p):
    return weighted_acc(s, y_t, y_p, count_color_variety)


def shape_weighted_accuracy(s, y_t, y_p):
    return weighted_acc(s, y_t, y_p, count_shape_variety)


def complexity_weighted_accuracy(s, y_t, y_p):
    return weighted_acc(
        s, y_t, y_p, lambda x: count_color_variety(x) * count_shape_variety(x)
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# ---------- Dataset (falls back to synthetic) ----------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("SPR_BENCH not found, making synthetic.")

    def synth(n):
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

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth(n)}, split="train"
        )

# ---------- Vectoriser ----------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])


def vec(lst: List[str]):
    return vectorizer.transform(lst).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vec, [dsets[s]["sequence"] for s in ["train", "dev", "test"]]
)
vocab_size = len(vectorizer.vocabulary_)
print("Vocab", vocab_size)

# ---------- Labels ----------
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], dtype=np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], dtype=np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], dtype=np.int64)
num_classes = len(labels)

batch_size = 64
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=batch_size,
)


# ---------- Model factory ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Hyper-parameter loop ----------
drop_rates = [0.0, 0.2, 0.4, 0.6]
for d in drop_rates:
    tag = f"dropout_{d}"
    print(f"\n=== Training with {tag} ===")
    model = MLP(vocab_size, num_classes, d).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"losses": {"train": [], "val": []}, "metrics": {"val": []}}
    epochs = 5
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)
        # validate
        model.eval()
        val_loss = 0.0
        preds = []
        tgts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                tgts.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        seq_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        cwa = color_weighted_accuracy(seq_val, tgts, preds)
        swa = shape_weighted_accuracy(seq_val, tgts, preds)
        compA = complexity_weighted_accuracy(seq_val, tgts, preds)
        hist["losses"]["train"].append(tr_loss)
        hist["losses"]["val"].append(val_loss)
        hist["metrics"]["val"].append(
            {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "compwa": compA}
        )
        print(f"Ep{ep}: tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} acc={acc:.3f}")
    # ---------- Test evaluation ----------
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        test_preds = test_logits.argmax(1).cpu().numpy()
    seq_test = dsets["test"]["sequence"]
    test_metrics = {
        "acc": (test_preds == y_test).mean(),
        "cwa": color_weighted_accuracy(seq_test, y_test, test_preds),
        "swa": shape_weighted_accuracy(seq_test, y_test, test_preds),
        "compwa": complexity_weighted_accuracy(seq_test, y_test, test_preds),
    }
    print("Test:", test_metrics)
    # store
    hist["predictions"] = test_preds
    hist["ground_truth"] = y_test
    hist["metrics"]["test"] = test_metrics
    experiment_data["dropout_rate"]["SPR_BENCH"][tag] = hist

# ---------- Save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved to", os.path.join(working_dir, "experiment_data.npy"))
