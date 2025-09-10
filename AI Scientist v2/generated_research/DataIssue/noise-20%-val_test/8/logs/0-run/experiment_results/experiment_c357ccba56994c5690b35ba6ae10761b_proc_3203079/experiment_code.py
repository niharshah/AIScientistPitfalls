import os
import pathlib
import random
import string
from typing import Dict, Any, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree

# ---------------- Saving helpers -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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
        seqs, labels = [], []
        for _ in range(n):
            s = gen_seq()
            seqs.append(s)
            labels.append(label_fn(s))
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

X_train_dense = X_train.toarray()
X_dev_dense = X_dev.toarray()
X_test_dense = X_test.toarray()


# --------------- SEFA helper ---------------------
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


def compute_sefa(clf: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray) -> float:
    preds = clf.predict(X)
    successes = 0
    for i in range(X.shape[0]):
        rule_pred = decision_tree_single_pred(clf, X[i])
        if rule_pred == preds[i] and rule_pred == y[i]:
            successes += 1
    return successes / X.shape[0]


# --------------- Hyper-parameter sweep -----------
depth_candidates: List[Any] = [3, 5, 7, 10, None]
experiment_data: Dict[str, Dict[str, Any]] = {
    "max_depth_tuning": {
        "SPR_BENCH": {
            "depths": depth_candidates,
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": [], "test": []},
            "sefa": [],
            "predictions": {},  # depth -> list
            "ground_truth": y_test.tolist(),
            "best_depth": None,
        }
    }
}

best_val_acc = -1.0
best_depth = None
best_clf = None

for depth in depth_candidates:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train_dense, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_dense))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev_dense))
    test_acc = accuracy_score(y_test, clf.predict(X_test_dense))

    experiment_data["max_depth_tuning"]["SPR_BENCH"]["metrics"]["train"].append(
        train_acc
    )
    experiment_data["max_depth_tuning"]["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["max_depth_tuning"]["SPR_BENCH"]["metrics"]["test"].append(test_acc)

    experiment_data["max_depth_tuning"]["SPR_BENCH"]["losses"]["train"].append(
        1 - train_acc
    )
    experiment_data["max_depth_tuning"]["SPR_BENCH"]["losses"]["val"].append(
        1 - val_acc
    )
    experiment_data["max_depth_tuning"]["SPR_BENCH"]["losses"]["test"].append(
        1 - test_acc
    )

    experiment_data["max_depth_tuning"]["SPR_BENCH"]["predictions"][str(depth)] = (
        clf.predict(X_test_dense).tolist()
    )
    experiment_data["max_depth_tuning"]["SPR_BENCH"]["sefa"].append(
        compute_sefa(clf, X_test_dense, y_test)
    )

    print(
        f"Depth={depth}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}"
    )

    if (val_acc > best_val_acc) or (
        val_acc == best_val_acc
        and (best_depth is None or (depth or 1e9) < (best_depth or 1e9))
    ):
        best_val_acc = val_acc
        best_depth = depth
        best_clf = clf

experiment_data["max_depth_tuning"]["SPR_BENCH"]["best_depth"] = best_depth
print(
    f"\nSelected best depth = {best_depth} with validation accuracy {best_val_acc:.4f}"
)

best_test_acc = accuracy_score(y_test, best_clf.predict(X_test_dense))
best_sefa = compute_sefa(best_clf, X_test_dense, y_test)
print(f"Best model test accuracy: {best_test_acc:.4f}")
print(f"Best model SEFA: {best_sefa:.4f}")

# --------------- Save artefacts ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir, "experiment_data.npy")}')
