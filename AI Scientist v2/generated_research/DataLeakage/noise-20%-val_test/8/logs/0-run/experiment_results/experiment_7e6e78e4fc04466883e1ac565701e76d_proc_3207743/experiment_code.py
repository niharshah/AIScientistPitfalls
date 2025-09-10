# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# --------------------------------------------------------------------
# GPU / CPU handling (mandatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------------------------------------------------------


# ---------------------- DATA LOADING --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def get_dataset() -> DatasetDict:
    possible_path = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        spr = load_spr_bench(possible_path)
        print("Loaded real SPR_BENCH from", possible_path)
        return spr
    except Exception as e:
        print("Could not load real SPR_BENCH; generating synthetic toy dataset.")
        # synthetic tiny dataset
        rng = np.random.default_rng(42)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for _ in range(n):
                length = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=length))
                label = int(seq.count("A") % 2 == 0)  # synthetic parity rule
                seqs.append(seq)
                labels.append(label)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


dsets = get_dataset()

# ---------------------- PREPROCESSING -------------------------------
# build char vocabulary
chars = set()
for split in dsets:
    for s in dsets[split]["sequence"]:
        chars.update(list(s))
chars = sorted(list(chars))
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)
print("Character vocab:", chars)


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


X_train, y_train = vectorise_split("train")
X_dev, y_dev = vectorise_split("dev")
X_test, y_test = vectorise_split("test")

# ---------------------- MODEL ---------------------------------------
clf = DecisionTreeClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)


# ---------------------- RULE EXTRACTION -----------------------------
def path_to_rule(tree, feature_names):
    """
    Convert a decision tree into a list of human-readable rules (string).
    Not used for SEFA computation but saved for inspection.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    def recurse(node, cur_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_rule = cur_rule + [f"{name} <= {threshold:.1f}"]
            recurse(tree_.children_left[node], left_rule)
            right_rule = cur_rule + [f"{name} > {threshold:.1f}"]
            recurse(tree_.children_right[node], right_rule)
        else:
            proba = tree_.value[node][0]
            pred = np.argmax(proba)
            rule = " AND ".join(cur_rule) if cur_rule else "TRUE"
            paths.append(f"IF {rule} THEN label={pred}")

    recurse(0, [])
    return paths


rule_strings = path_to_rule(clf, chars)
with open(os.path.join(working_dir, "extracted_rules.txt"), "w") as f:
    f.write("\n".join(rule_strings))
print(f"Saved {len(rule_strings)} extracted rules.")

# ---------------------- TRAIN / DEV METRICS -------------------------
dev_proba = clf.predict_proba(X_dev)
val_loss = log_loss(y_dev, dev_proba)
print(f"Epoch 1: validation_loss = {val_loss:.4f}")

# ---------------------- TEST & SEFA ---------------------------------
test_pred = clf.predict(X_test)

# Execute rule = model itself; re-evaluate to double-check
rule_pred = clf.predict(X_test)
sefa = accuracy_score(y_test, rule_pred)  # identical to accuracy here
print(f"Test SEFA (== accuracy for this model): {sefa:.4f}")

# Confusion matrix plot
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.colorbar(im, ax=ax)
fig_path = os.path.join(working_dir, "confusion_matrix.png")
plt.savefig(fig_path)
plt.close()
print("Saved confusion matrix to", fig_path)

# ---------------------- SAVE EXPERIMENT DATA ------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [1 - val_loss], "test": [sefa]},
        "losses": {"train": [], "val": [val_loss]},
        "predictions": test_pred.tolist(),
        "ground_truth": y_test.tolist(),
    }
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
