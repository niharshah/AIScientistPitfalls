import os, pathlib, random, string, numpy as np, torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from datasets import DatasetDict, Dataset
from typing import Dict

# ----------------- Repro/paths -------------------
np.random.seed(42)
random.seed(42)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- Device info -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- Data utils --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def create_synthetic_spr(n_train=2000, n_dev=500, n_test=500) -> DatasetDict:
    def gen_seq():
        return "".join(
            random.choices(
                list(string.ascii_uppercase) + list(string.digits),
                k=random.randint(6, 12),
            )
        )

    def label_fn(seq):
        return 1 if seq.count("A") > seq.count("B") else 0

    def build(n):
        seqs = [gen_seq() for _ in range(n)]
        return Dataset.from_dict(
            {
                "id": list(range(n)),
                "sequence": seqs,
                "label": [label_fn(s) for s in seqs],
            }
        )

    return DatasetDict(
        {"train": build(n_train), "dev": build(n_dev), "test": build(n_test)}
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
data = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else create_synthetic_spr()
print({k: len(v) for k, v in data.items()})

# ----------------- Vectorizer --------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
X_train = vectorizer.fit_transform(data["train"]["sequence"]).toarray()
y_train = np.array(data["train"]["label"])
X_dev = vectorizer.transform(data["dev"]["sequence"]).toarray()
y_dev = np.array(data["dev"]["label"])
X_test = vectorizer.transform(data["test"]["sequence"]).toarray()
y_test = np.array(data["test"]["label"])

# --------------- Experiment tracker --------------
experiment_data = {
    "min_samples_leaf_tuning": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "param_values": [],
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "best_param": None,
            "best_val_acc": None,
        }
    }
}


# --------------- Helper for SEFA -----------------
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
        if decision_tree_single_pred(clf, X[i]) == preds[i] == y[i]:
            successes += 1
    return successes / X.shape[0]


# --------------- Hyperparameter tuning ----------
param_grid = [1, 5, 10, 20]
best_clf, best_val_acc, best_param = None, -1.0, None
for p in param_grid:
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=p, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev))
    experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["metrics"]["train"].append(
        train_acc
    )
    experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["metrics"]["val"].append(
        val_acc
    )
    experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["losses"]["train"].append(
        1 - train_acc
    )
    experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["losses"]["val"].append(
        1 - val_acc
    )
    experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["param_values"].append(p)
    print(f"min_samples_leaf={p:2d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc, best_clf, best_param = val_acc, clf, p

experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["best_param"] = best_param
experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["best_val_acc"] = best_val_acc
print(f"Best min_samples_leaf = {best_param} with dev accuracy {best_val_acc:.4f}")

# --------------- Test evaluation ----------------
test_acc = accuracy_score(y_test, best_clf.predict(X_test))
sefa = compute_sefa(best_clf, X_test, y_test)
experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["predictions"] = (
    best_clf.predict(X_test).tolist()
)
experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["test_acc"] = test_acc
experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]["SEFA"] = sefa
print(f"Test accuracy: {test_acc:.4f}")
print(f"Self-Explain Fidelity Accuracy (SEFA): {sefa:.4f}")

# --------------- Save artefacts -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f'Experiment data saved to {os.path.join(working_dir,"experiment_data.npy")}')
