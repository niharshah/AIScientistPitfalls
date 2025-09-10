import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ----------------------- I/O -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "learning_rate": {  # hyper-parameter being tuned
        "SPR_BENCH": {"runs": {}}  # dataset name  # will store one entry per lr value
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------- Utils ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for sp in ["train", "dev", "test"]:
        out[sp] = _load(f"{sp}.csv")
    return out


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ------------------- Data --------------------------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("SPR_BENCH not found, generating synthetic dataset.")

    def synth(n):
        seqs, labels = [], []
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        for _ in range(n):
            seq = " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            labels.append(np.random.choice(["A", "B", "C"]))
            seqs.append(seq)
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for sp, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[sp] = load_dataset("json", data_files={"train": synth(n)}, split="train")

# Vectoriser
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])
vocab_size = len(vectorizer.vocabulary_)
print("Vocabulary size:", vocab_size)


def vectorize(seqs: List[str]) -> np.ndarray:
    return vectorizer.transform(seqs).toarray().astype(np.float32)


X_train = vectorize(dsets["train"]["sequence"])
X_val = vectorize(dsets["dev"]["sequence"])
X_test = vectorize(dsets["test"]["sequence"])

labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
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


# ------------------- Model -------------------------
class MLP(nn.Module):
    def __init__(self, inp, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 256), nn.ReLU(), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.net(x)


# ---------------- Hyper-parameter sweep ------------
lrs = [3e-4, 5e-4, 1e-3, 2e-3]
epochs = 5

for lr in lrs:
    print(f"\n=== Training with learning_rate={lr} ===")
    run_key = f"lr_{lr}"
    experiment_data["learning_rate"]["SPR_BENCH"]["runs"][run_key] = {
        "losses": {"train": [], "val": []},
        "metrics": {"val": [], "test": {}},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }

    model = MLP(vocab_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------- training loop -------
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

        # validation
        model.eval()
        val_loss = 0.0
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                tgts.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        cwa = color_weighted_accuracy(seqs_val, tgts, preds)
        swa = shape_weighted_accuracy(seqs_val, tgts, preds)
        comp = complexity_weighted_accuracy(seqs_val, tgts, preds)

        # logging
        run_store = experiment_data["learning_rate"]["SPR_BENCH"]["runs"][run_key]
        run_store["losses"]["train"].append(tr_loss)
        run_store["losses"]["val"].append(val_loss)
        run_store["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )

        print(
            f"Epoch {epoch} | TrainLoss {tr_loss:.4f} ValLoss {val_loss:.4f} "
            f"ACC {acc:.3f} CWA {cwa:.3f} SWA {swa:.3f} CompWA {comp:.3f}"
        )

    # -------- Test evaluation --------
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        test_preds = logits.argmax(1).cpu().numpy()
    test_seqs = dsets["test"]["sequence"]
    test_acc = (test_preds == y_test).mean()
    test_cwa = color_weighted_accuracy(test_seqs, y_test, test_preds)
    test_swa = shape_weighted_accuracy(test_seqs, y_test, test_preds)
    test_comp = complexity_weighted_accuracy(test_seqs, y_test, test_preds)

    run_store["metrics"]["test"] = {
        "acc": test_acc,
        "cwa": test_cwa,
        "swa": test_swa,
        "compwa": test_comp,
    }
    run_store["predictions"] = test_preds
    run_store["ground_truth"] = y_test
    run_store["sequences"] = test_seqs

    print(
        f"Test  ACC {test_acc:.3f} CWA {test_cwa:.3f} "
        f"SWA {test_swa:.3f} CompWA {test_comp:.3f}"
    )

# ------------------ Save everything -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nSaved all results to {os.path.join(working_dir,'experiment_data.npy')}")
