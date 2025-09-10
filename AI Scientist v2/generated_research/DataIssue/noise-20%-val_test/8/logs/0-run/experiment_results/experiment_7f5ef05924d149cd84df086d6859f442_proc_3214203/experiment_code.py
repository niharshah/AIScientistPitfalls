# Length-Normalized Feature Ablation for Parity Task
import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# ---------------------- I/O & ENV -----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- DATA LOADING --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def get_dataset() -> DatasetDict:
    try:
        path = pathlib.Path(os.getcwd()) / "SPR_BENCH"
        dsets = load_spr_bench(path)
        print("Loaded real SPR_BENCH")
        return dsets
    except Exception:
        # synthetic parity data
        rng, vocab = np.random.default_rng(42), list("ABC")

        def gen(n):
            seqs, labels = [], []
            for _ in range(n):
                length = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=length))
                labels.append(int(seq.count("A") % 2 == 0))
                seqs.append(seq)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


dsets = get_dataset()

# ---------------------- VOCAB & VECTORIZATION -----------------------
chars = sorted({ch for split in dsets for s in dsets[split]["sequence"] for ch in s})
char2idx, V = {c: i for i, c in enumerate(chars)}, len(chars)
print("Character vocab:", chars)


def seq_to_vec(seq: str, normalize=False) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    for ch in seq:
        if ch in char2idx:
            v[char2idx[ch]] += 1.0
    if normalize and v.sum() > 0:
        v /= v.sum()
    return v


def vectorise_split(split, normalize=False):
    X = np.stack([seq_to_vec(s, normalize) for s in dsets[split]["sequence"]])
    y = np.array(dsets[split]["label"])
    return X, y


# ---------------------- TRAIN / EVAL HELPER -------------------------
def train_and_eval(name, normalize=False):
    X_tr, y_tr = vectorise_split("train", normalize)
    X_dev, y_dev = vectorise_split("dev", normalize)
    X_te, y_te = vectorise_split("test", normalize)

    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_tr, y_tr)

    dev_proba = clf.predict_proba(X_dev)
    val_loss = log_loss(y_dev, dev_proba)
    test_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, test_pred)

    # Confusion matrix plot
    cm = confusion_matrix(y_te, test_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xlabel="Predicted", ylabel="True", title=f"{name} Confusion Matrix")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(working_dir, f"cm_{name}.png"))
    plt.close()

    return {
        "metrics": {"train": [], "val": [1 - val_loss], "test": [acc]},
        "losses": {"train": [], "val": [val_loss]},
        "predictions": test_pred.tolist(),
        "ground_truth": y_te.tolist(),
    }


# ---------------------- RUN BASELINE & ABLATION ---------------------
experiment_data = {
    "baseline": {"SPR_BENCH": train_and_eval("baseline", False)},
    "length_normalized": {"SPR_BENCH": train_and_eval("length_norm", True)},
}

# ---------------------- SAVE ----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
