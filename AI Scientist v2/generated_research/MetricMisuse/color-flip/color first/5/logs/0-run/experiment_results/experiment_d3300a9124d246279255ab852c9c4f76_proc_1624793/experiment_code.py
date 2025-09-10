import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# -------------  Set-up & bookkeeping -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "weight_decay": {"SPR_BENCH": {}}  # hyper-parameter tuning type  # dataset
}


# -------------  Metrics helpers -------------
def count_color_variety(seq: str):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str):
    return len(set(tok[0] for tok in seq.split() if tok))


def _wa(seqs, y_true, y_pred, w_fn):
    w = [w_fn(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(s, y_t, y_p):
    return _wa(s, y_t, y_p, count_color_variety)


def shape_weighted_accuracy(s, y_t, y_p):
    return _wa(s, y_t, y_p, count_shape_variety)


def complexity_weighted_accuracy(s, y_t, y_p):
    return _wa(
        s, y_t, y_p, lambda seq: count_color_variety(seq) * count_shape_variety(seq)
    )


# -------------  Load SPR-BENCH or synthetic -------------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path):
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({spl: _ld(f"{spl}.csv") for spl in ["train", "dev", "test"]})


dataset_available = pathlib.Path(DATA_ENV).exists()
if dataset_available:
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("SPR_BENCH not found – generating synthetic data.")

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

    dsets = DatasetDict()
    for spl, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[spl] = load_dataset(
            "json", data_files={"train": synth_split(n)}, split="train"
        )

# -------------  Vectorisation -------------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])
vocab_size = len(vectorizer.vocabulary_)


def vec(seqs: List[str]):
    return vectorizer.transform(seqs).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vec, [dsets[s]["sequence"] for s in ["train", "dev", "test"]]
)

# -------------  Label encoding -------------
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], np.int64)
num_classes = len(labels)

# -------------  DataLoaders -------------
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


# -------------  Model definition -------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# -------------  Hyper-parameter sweep -------------
weight_decay_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
epochs = 5
criterion = nn.CrossEntropyLoss()

for wd in weight_decay_values:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = MLP(vocab_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    run_data = {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "sequences": dsets["test"]["sequence"],
        "weight_decay": wd,
    }

    for epoch in range(1, epochs + 1):
        # --- training ---
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
        run_data["losses"]["train"].append(tr_loss)

        # --- validation ---
        model.eval()
        val_loss, all_preds, all_tgts = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_tgts.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        run_data["losses"]["val"].append(val_loss)

        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(all_preds) == np.array(all_tgts)).mean()
        cwa = color_weighted_accuracy(seqs_val, all_tgts, all_preds)
        swa = shape_weighted_accuracy(seqs_val, all_tgts, all_preds)
        comp = complexity_weighted_accuracy(seqs_val, all_tgts, all_preds)
        run_data["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )

    # --- Test evaluation ---
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        test_preds = logits.argmax(1).cpu().numpy()
    test_acc = (test_preds == y_test).mean()
    test_cwa = color_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    test_swa = shape_weighted_accuracy(dsets["test"]["sequence"], y_test, test_preds)
    test_comp = complexity_weighted_accuracy(
        dsets["test"]["sequence"], y_test, test_preds
    )
    run_data["metrics"]["test"] = {
        "acc": test_acc,
        "cwa": test_cwa,
        "swa": test_swa,
        "compwa": test_comp,
    }
    run_data["predictions"] = test_preds.tolist()

    print(
        f"Test: ACC={test_acc:.3f} CWA={test_cwa:.3f} SWA={test_swa:.3f} CompWA={test_comp:.3f}"
    )

    # store run
    experiment_data["weight_decay"]["SPR_BENCH"][str(wd)] = run_data

# -------------  Save everything -------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nSaved all results to {os.path.join(working_dir,'experiment_data.npy')}")
