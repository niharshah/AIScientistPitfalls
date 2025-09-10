import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List

# ---------- experiment dict ----------
experiment_data = {
    "hidden_dim_size": {
        "SPR_BENCH": {}  # a sub-dict per hidden size will be filled later
    }
}

# ---------- basic setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
print(f"Using device: {device}")


# ---------- SPR-BENCH loader ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for sp in ["train", "dev", "test"]:
        dd[sp] = _load(f"{sp}.csv")
    return dd


DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("SPR_BENCH not found. Generating synthetic toy data.")

    def synth_split(n):
        seqs, labels = [], []
        shapes, colors, labs = ["▲", "●", "■"], ["r", "g", "b"], ["A", "B", "C"]
        for _ in range(n):
            seq = " ".join(
                np.random.choice(
                    [s + c for s in shapes for c in colors],
                    size=np.random.randint(3, 8),
                )
            )
            seqs.append(seq)
            labels.append(np.random.choice(labs))
        return {"sequence": seqs, "label": labels}

    dsets = DatasetDict()
    for sp, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[sp] = load_dataset(
            "json", data_files={"train": synth_split(n)}, split="train"
        )


# ---------- helper metrics ----------
def count_color_variety(seq: str):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------- vectorisation ----------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
vectorizer.fit(dsets["train"]["sequence"])


def vectorize(seqs: List[str]) -> np.ndarray:
    return vectorizer.transform(seqs).toarray().astype(np.float32)


X_train, X_val, X_test = map(
    vectorize,
    (dsets["train"]["sequence"], dsets["dev"]["sequence"], dsets["test"]["sequence"]),
)

# ---------- labels ----------
labels_sorted = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels_sorted)}
id2label = {i: l for l, i in label2id.items()}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], np.int64)

# ---------- DataLoaders (shared) ----------
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


# ---------- model factory ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------- hyperparameter grid ----------
hidden_sizes = [64, 128, 256, 512, 1024]
epochs = 5
input_dim, num_classes = X_train.shape[1], len(labels_sorted)

for hsize in hidden_sizes:
    print(f"\n=== Training with hidden_dim={hsize} ===")
    model = MLP(input_dim, hsize, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_train_loss, log_val_loss, log_val_metrics = [], [], []

    # ---- epochs ----
    for ep in range(1, epochs + 1):
        # Training
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        log_train_loss.append(train_loss)

        # Validation
        model.eval()
        vloss, preds, tgts = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vloss += criterion(logits, yb).item() * xb.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                tgts.extend(yb.cpu().numpy())
        val_loss = vloss / len(val_loader.dataset)
        log_val_loss.append(val_loss)

        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        cwa = color_weighted_accuracy(seqs_val, tgts, preds)
        swa = shape_weighted_accuracy(seqs_val, tgts, preds)
        comp = complexity_weighted_accuracy(seqs_val, tgts, preds)
        log_val_metrics.append(
            {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )
        print(
            f"Epoch {ep}/{epochs}  TLoss={train_loss:.4f}  VLoss={val_loss:.4f}  "
            f"ACC={acc:.3f}  CWA={cwa:.3f}  SWA={swa:.3f}  CompWA={comp:.3f}"
        )

    # ---- final test ----
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        test_preds = test_logits.argmax(1).cpu().numpy()
    seqs_test = dsets["test"]["sequence"]
    test_acc = (test_preds == y_test).mean()
    test_cwa = color_weighted_accuracy(seqs_test, y_test, test_preds)
    test_swa = shape_weighted_accuracy(seqs_test, y_test, test_preds)
    test_comp = complexity_weighted_accuracy(seqs_test, y_test, test_preds)
    test_metrics = {
        "acc": test_acc,
        "cwa": test_cwa,
        "swa": test_swa,
        "compwa": test_comp,
    }
    print(
        f"Test  ACC={test_acc:.3f}  CWA={test_cwa:.3f}  "
        f"SWA={test_swa:.3f}  CompWA={test_comp:.3f}"
    )

    # ---- save into experiment_data ----
    experiment_data["hidden_dim_size"]["SPR_BENCH"][str(hsize)] = {
        "losses": {"train": log_train_loss, "val": log_val_loss},
        "metrics": {"val": log_val_metrics, "test": test_metrics},
        "predictions": test_preds,
        "ground_truth": y_test,
        "sequences": seqs_test,
    }

# ---------- persist ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nSaved all results to {os.path.join(working_dir,'experiment_data.npy')}")
