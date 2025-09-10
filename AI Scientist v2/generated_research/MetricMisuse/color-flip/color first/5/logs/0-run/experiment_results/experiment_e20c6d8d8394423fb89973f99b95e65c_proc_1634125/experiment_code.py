import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict
from typing import List, Dict

# ---------- Paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Experiment container ----------
experiment_data: Dict = {"activation_function": {"SPR_BENCH": {}}}


# ---------- SPR-BENCH helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------- Load / synthesize data ----------
DATA_ENV = os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if pathlib.Path(DATA_ENV).exists():
    dsets = load_spr_bench(pathlib.Path(DATA_ENV))
else:
    print("SPR_BENCH not found → using small synthetic demo data.")
    shapes, colors = ["▲", "●", "■"], ["r", "g", "b"]

    def synth(n):
        seqs, lbls = [], []
        for _ in range(n):
            seqs.append(
                " ".join(
                    np.random.choice(
                        [s + c for s in shapes for c in colors],
                        size=np.random.randint(3, 8),
                    )
                )
            )
            lbls.append(np.random.choice(["A", "B", "C"]))
        return {"sequence": seqs, "label": lbls}

    dsets = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [200, 50, 50]):
        dsets[split] = load_dataset(
            "json", data_files={"train": synth(n)}, split="train"
        )

# ---------- Vectorizer ----------
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

# ---------- Labels ----------
labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
y_train = np.array([label2id[l] for l in dsets["train"]["label"]], np.int64)
y_val = np.array([label2id[l] for l in dsets["dev"]["label"]], np.int64)
y_test = np.array([label2id[l] for l in dsets["test"]["label"]], np.int64)
num_classes = len(labels)
print(f"Number of classes: {num_classes}")

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


# ---------- Model definition ----------
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, act_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), act_layer(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------- Training routine ----------
def run_experiment(act_name: str, act_cls):
    torch.manual_seed(0)
    np.random.seed(0)
    model = MLP(vocab_size, num_classes, act_cls).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data_store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }

    epochs = 5
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        data_store["losses"]["train"].append(train_loss)

        # --- Validate ---
        model.eval()
        val_loss, preds, tgts = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                tgts.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        data_store["losses"]["val"].append(val_loss)

        seqs_val = dsets["dev"]["sequence"]
        acc = (np.array(preds) == np.array(tgts)).mean()
        cwa = color_weighted_accuracy(seqs_val, tgts, preds)
        swa = shape_weighted_accuracy(seqs_val, tgts, preds)
        comp = complexity_weighted_accuracy(seqs_val, tgts, preds)
        data_store["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "compwa": comp}
        )
        print(
            f"[{act_name}] Epoch {epoch} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )

    # --- Test ---
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        test_preds = logits.argmax(1).cpu().numpy()
    seqs_test = dsets["test"]["sequence"]
    test_metrics = {
        "acc": (test_preds == y_test).mean(),
        "cwa": color_weighted_accuracy(seqs_test, y_test, test_preds),
        "swa": shape_weighted_accuracy(seqs_test, y_test, test_preds),
        "compwa": complexity_weighted_accuracy(seqs_test, y_test, test_preds),
    }
    data_store["predictions"] = test_preds
    data_store["ground_truth"] = y_test
    data_store["sequences"] = seqs_test
    data_store["metrics"]["test"] = test_metrics
    print(
        f"[{act_name}] Test — ACC={test_metrics['acc']:.3f} "
        f"CWA={test_metrics['cwa']:.3f} SWA={test_metrics['swa']:.3f} "
        f"CompWA={test_metrics['compwa']:.3f}\n"
    )
    return data_store


# ---------- Activation sweep ----------
activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": lambda: nn.LeakyReLU(0.01),
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "SELU": nn.SELU,
}

for act_name, act_cls in activations.items():
    experiment_data["activation_function"]["SPR_BENCH"][act_name] = run_experiment(
        act_name, act_cls
    )

# ---------- Save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved results to {os.path.join(working_dir,'experiment_data.npy')}")
