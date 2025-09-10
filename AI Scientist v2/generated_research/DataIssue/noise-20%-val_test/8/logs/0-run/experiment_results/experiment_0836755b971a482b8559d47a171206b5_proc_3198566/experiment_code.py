import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import random
import string
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from datasets import DatasetDict, Dataset
import pathlib
from typing import Dict

# ---------------- Device handling ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Data loading -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

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


def create_synthetic_spr(n_train=2000, n_dev=500, n_test=500) -> DatasetDict:
    def gen_seq():
        length = random.randint(6, 12)
        return "".join(
            random.choices(list(string.ascii_uppercase) + list(string.digits), k=length)
        )

    def label_fn(seq):
        # simple synthetic rule: label 1 if count('A') > count('B') else 0
        return 1 if seq.count("A") > seq.count("B") else 0

    def build(n):
        seqs = [gen_seq() for _ in range(n)]
        labels = [label_fn(s) for s in seqs]
        ids = list(range(n))
        return Dataset.from_dict({"id": ids, "sequence": seqs, "label": labels})

    return DatasetDict(
        {"train": build(n_train), "dev": build(n_dev), "test": build(n_test)}
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    print("Loading real SPR_BENCH dataset...")
    data = load_spr_bench(DATA_PATH)
else:
    print("Real dataset not found, falling back to synthetic data.")
    data = create_synthetic_spr()

print({k: len(v) for k, v in data.items()})

# --------------- Vectorization -------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
X_train = vectorizer.fit_transform(data["train"]["sequence"])
y_train = np.array(data["train"]["label"])
X_dev = vectorizer.transform(data["dev"]["sequence"])
y_dev = np.array(data["dev"]["label"])
X_test = vectorizer.transform(data["test"]["sequence"])
y_test = np.array(data["test"]["label"])

# --------------- Model training ------------------
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train.toarray(), y_train)  # small enough to densify


# --------------- Evaluation helpers --------------
def decision_tree_single_pred(clf: DecisionTreeClassifier, x_row: np.ndarray) -> int:
    tree = clf.tree_
    node = 0
    while tree.children_left[node] != _tree.TREE_LEAF:
        feat = tree.feature[node]
        thresh = tree.threshold[node]
        node = (
            tree.children_left[node]
            if x_row[feat] <= thresh
            else tree.children_right[node]
        )
    return np.argmax(tree.value[node][0])


def compute_sefa(clf, X, y):
    preds = clf.predict(X)
    successes = 0
    for i in range(X.shape[0]):
        rule_pred = decision_tree_single_pred(clf, X[i])
        if rule_pred == preds[i] and rule_pred == y[i]:
            successes += 1
    return successes / X.shape[0]


# --------------- Metrics tracking ----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

train_acc = accuracy_score(y_train, clf.predict(X_train))
val_acc = accuracy_score(y_dev, clf.predict(X_dev))
train_loss = 1 - train_acc
val_loss = 1 - val_acc
experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_acc)
experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)
experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

print(f"Epoch 0: validation_loss = {val_loss:.4f} (val_accuracy = {val_acc:.4f})")

# --------------- Test & SEFA ---------------------
test_acc = accuracy_score(y_test, clf.predict(X_test))
sefa = compute_sefa(clf, X_test.toarray(), y_test)
experiment_data["SPR_BENCH"]["predictions"] = clf.predict(X_test).tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = y_test.tolist()

print(f"Test accuracy: {test_acc:.4f}")
print(f"Self-Explain Fidelity Accuracy (SEFA): {sefa:.4f}")

# --------------- Save artefacts ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir, "experiment_data.npy")}')
