import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ---------- bookkeeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "batch_size": {  # hyper-parameter we sweep
        "SPR_BENCH": {"runs": []}  # list of per-batch-size dicts
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper functions ----------
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


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- data ----------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("SPR_BENCH not found – generating synthetic data.")

    def synth_split(n):
        shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]
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

labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], dtype=np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], dtype=np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], dtype=np.int64)
num_classes = len(labels)
print("Classes:", labels)


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------- hyper-parameter sweep ----------
batch_sizes = [16, 32, 64, 128]
epochs = 5
for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    run_record = {
        "batch_size": bs,
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test,
        "sequences": dsets["test"]["sequence"],
    }

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=bs,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=bs
    )

    # fresh model
    model = MLP(vocab_size, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        run_record["losses"]["train"].append(train_loss)

        # validation
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
        run_record["losses"]["val"].append(val_loss)

        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        cwa = color_weighted_accuracy(seqs_val, tgts, preds)
        swa = shape_weighted_accuracy(seqs_val, tgts, preds)
        comp = complexity_weighted_accuracy(seqs_val, tgts, preds)
        run_record["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )
        print(
            f"Epoch {epoch}: loss {train_loss:.3f}/{val_loss:.3f} "
            f"ACC {acc:.3f} CWA {cwa:.3f} SWA {swa:.3f} CompWA {comp:.3f}"
        )

    # final test evaluation
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        test_preds = logits.argmax(1).cpu().numpy()
    tacc = (test_preds == y_test).mean()
    tcwa = color_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    tswa = shape_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    tcomp = complexity_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    run_record["metrics"]["test"] = {
        "acc": tacc,
        "cwa": tcwa,
        "swa": tswa,
        "compwa": tcomp,
    }
    run_record["predictions"] = test_preds
    print(f"Test: ACC {tacc:.3f} CWA {tcwa:.3f} SWA {tswa:.3f} CompWA {tcomp:.3f}")

    # store
    experiment_data["batch_size"]["SPR_BENCH"]["runs"].append(run_record)

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
