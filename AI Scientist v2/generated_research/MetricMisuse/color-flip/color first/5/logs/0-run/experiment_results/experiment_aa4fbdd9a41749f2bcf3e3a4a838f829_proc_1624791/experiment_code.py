import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ------------------- I/O & bookkeeping -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {"runs": []}  # each element is a dict with info for one lr value
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


# ------------------- helper functions --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


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


# ------------------- load dataset ------------------------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:  # synthetic fallback
    print("SPR_BENCH not found -> generating synthetic demo data.")

    def synth_split(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
        seq, lab = [], []
        for _ in range(n):
            seq.append(
                " ".join(
                    np.random.choice(
                        [s + c for s in shapes for c in colors],
                        size=np.random.randint(3, 8),
                    )
                )
            )
            lab.append(np.random.choice(["A", "B", "C"]))
        return {"sequence": seq, "label": lab}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth_split(n)}, split="train"
        )

# ------------------- vectorise text ----------------------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])
vocab_size = len(vectorizer.vocabulary_)
print("Vocabulary size:", vocab_size)


def vectorize(seqs: List[str]) -> np.ndarray:
    return vectorizer.transform(seqs).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vectorize,
    (dsets["train"]["sequence"], dsets["dev"]["sequence"], dsets["test"]["sequence"]),
)

# ------------------- labels ------------------------------
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], np.int64)
num_classes = len(labels)

# ------------------- data loaders ------------------------
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


# ------------------- model def ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ------------------- LR SWEEP ----------------------------
lr_grid = [1e-4, 3e-4, 1e-3, 3e-3]
epochs = 5

for lr in lr_grid:
    print(f"\n=== Training with learning rate = {lr:.0e} ===")
    model = MLP(vocab_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_data = {
        "lr": lr,
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }

    for epoch in range(1, epochs + 1):
        # ---- train pass ----
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        run_data["losses"]["train"].append(train_loss)
        run_data["metrics"]["train"].append({"epoch": epoch, "acc": train_acc})

        # ---- validation ----
        model.eval()
        val_loss, preds, tgts = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                tgts.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        run_data["losses"]["val"].append(val_loss)

        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        cwa = color_weighted_accuracy(seqs_val, tgts, preds)
        swa = shape_weighted_accuracy(seqs_val, tgts, preds)
        comp = complexity_weighted_accuracy(seqs_val, tgts, preds)
        run_data["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )

    # ------------- test pass ---------------
    with torch.no_grad():
        logits_test = model(torch.from_numpy(X_test).to(device))
        test_preds = logits_test.argmax(1).cpu().numpy()
    test_seqs = dsets["test"]["sequence"]
    test_acc = (test_preds == y_test).mean()
    test_cwa = color_weighted_accuracy(test_seqs, y_test, test_preds)
    test_swa = shape_weighted_accuracy(test_seqs, y_test, test_preds)
    test_comp = complexity_weighted_accuracy(test_seqs, y_test, test_preds)
    run_data["metrics"]["test"] = {
        "acc": test_acc,
        "cwa": test_cwa,
        "swa": test_swa,
        "compwa": test_comp,
    }
    run_data["predictions"] = test_preds
    run_data["ground_truth"] = y_test
    run_data["sequences"] = test_seqs

    print(
        f"Test — ACC={test_acc:.3f} CWA={test_cwa:.3f} "
        f"SWA={test_swa:.3f} CompWA={test_comp:.3f}"
    )

    experiment_data["learning_rate"]["SPR_BENCH"]["runs"].append(run_data)

# ------------------- save everything ---------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
