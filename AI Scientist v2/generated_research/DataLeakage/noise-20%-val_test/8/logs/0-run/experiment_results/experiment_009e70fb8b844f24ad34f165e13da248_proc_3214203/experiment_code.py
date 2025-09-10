import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch, random
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset

# -------------------- ENV / FOLDERS --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
rng = np.random.default_rng(0)
random.seed(0)


# -------------------- DATASET --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # small helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for split in ["train", "dev", "test"]:
        dd[split] = _load(f"{split}.csv")
    return dd


def get_dataset() -> DatasetDict:
    path = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        ds = load_spr_bench(path)
        print("Loaded real SPR_BENCH from", path)
        return ds
    except Exception:
        print("Could not load real SPR_BENCH; generating synthetic data.")
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                L = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=L))
                labels.append(int(seq.count("A") % 2 == 0))
                seqs.append(seq)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


dsets = get_dataset()

# -------------------- VECTORIZATION --------------------
chars = sorted({c for split in dsets for s in dsets[split]["sequence"] for c in s})
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)


def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    for ch in seq:
        if ch in char2idx:
            v[char2idx[ch]] += 1.0
    return v


def vectorise_split(split_name):
    X = np.stack([seq_to_vec(s) for s in dsets[split_name]["sequence"]])
    y = np.array(dsets[split_name]["label"])
    return X, y


X_train_full, y_train_full = vectorise_split("train")
X_dev, y_dev = vectorise_split("dev")
X_test, y_test = vectorise_split("test")

# -------------------- ABLATION CONFIG ------------------
fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
val_acc_list, test_acc_list, val_loss_list = [], [], []
predictions_dict = {}  # fraction -> list of preds

# -------------------- TRAIN / EVAL LOOP ----------------
for frac in fractions:
    # Stratified subsample
    if frac < 1.0:
        X_sub, _, y_sub, _ = train_test_split(
            X_train_full,
            y_train_full,
            train_size=frac,
            random_state=0,
            stratify=y_train_full,
        )
    else:
        X_sub, y_sub = X_train_full, y_train_full

    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_sub, y_sub)

    dev_proba = clf.predict_proba(X_dev)
    v_loss = log_loss(y_dev, dev_proba)
    v_pred = np.argmax(dev_proba, axis=1)
    v_acc = accuracy_score(y_dev, v_pred)

    t_pred = clf.predict(X_test)
    t_acc = accuracy_score(y_test, t_pred)  # == SEFA

    val_acc_list.append(v_acc)
    test_acc_list.append(t_acc)
    val_loss_list.append(v_loss)
    predictions_dict[str(frac)] = t_pred.tolist()

    print(
        f"Fraction {frac:.2f} | Dev acc {v_acc:.4f} | Test acc {t_acc:.4f} | Val loss {v_loss:.4f}"
    )

# -------------------- PLOTS ----------------------------
plt.figure(figsize=(5, 3))
plt.plot(fractions, val_acc_list, "o-", label="Validation acc")
plt.plot(fractions, test_acc_list, "s-", label="Test / SEFA acc")
plt.xlabel("Training fraction")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Training Data Size")
plt.legend()
plot_path = os.path.join(working_dir, "accuracy_vs_data_fraction.png")
plt.savefig(plot_path)
plt.close()
print("Saved plot to", plot_path)

# -------------------- CONFUSION MATRIX (full data) ----
clf_full = DecisionTreeClassifier(max_depth=5, random_state=0).fit(
    X_train_full, y_train_full
)
cm = confusion_matrix(y_test, clf_full.predict(X_test))
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (100%)")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.colorbar(im, ax=ax)
cm_path = os.path.join(working_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print("Saved confusion matrix to", cm_path)

# -------------------- SAVE EXPERIMENT DATA -------------
experiment_data = {
    "training_data_size_ablation": {
        "SPR_BENCH": {
            "fractions": fractions,
            "metrics": {
                "val_accuracy": val_acc_list,
                "test_accuracy": test_acc_list,
            },
            "losses": {
                "val_logloss": val_loss_list,
            },
            "predictions": predictions_dict,
            "ground_truth": y_test.tolist(),
        }
    }
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
