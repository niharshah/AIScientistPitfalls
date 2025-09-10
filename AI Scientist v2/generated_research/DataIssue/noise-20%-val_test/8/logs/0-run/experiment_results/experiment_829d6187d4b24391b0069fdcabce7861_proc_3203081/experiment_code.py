import os, pathlib, random, string, numpy as np, torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict

# ---------- I/O setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Device (informational only) ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- Data helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
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

    def label_fn(seq):  # simple rule
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


# Try to load real dataset, otherwise create synthetic
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
data = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else create_synthetic_spr()
print({split: len(ds) for split, ds in data.items()})

# ---------- Vectorisation ----------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
X_train = vectorizer.fit_transform(data["train"]["sequence"])
X_dev = vectorizer.transform(data["dev"]["sequence"])
X_test = vectorizer.transform(data["test"]["sequence"])
y_train = np.array(data["train"]["label"])
y_dev = np.array(data["dev"]["label"])
y_test = np.array(data["test"]["label"])


# ---------- SEFA helper ----------
def decision_tree_single_pred(clf: DecisionTreeClassifier, x_row: np.ndarray) -> int:
    tree = clf.tree_
    node = 0
    while tree.children_left[node] != _tree.TREE_LEAF:
        feat, thresh = tree.feature[node], tree.threshold[node]
        node = (
            tree.children_left[node]
            if x_row[feat] <= thresh
            else tree.children_right[node]
        )
    return np.argmax(tree.value[node][0])


def compute_sefa(clf, X_dense: np.ndarray, y_true: np.ndarray) -> float:
    preds = clf.predict(X_dense)
    successes = 0
    for i in range(X_dense.shape[0]):
        rule_pred = decision_tree_single_pred(clf, X_dense[i])
        if rule_pred == preds[i] == y_true[i]:
            successes += 1
    return successes / X_dense.shape[0]


# ---------- Hyperparameter tuning ----------
param_grid = [2, 4, 8, 16, 32]
best_clf, best_val_acc, best_nodes = None, -1.0, 1e9

experiment_data = {
    "min_samples_split": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "param_values": param_grid,
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "best_param": None,
        }
    }
}

for idx, mss in enumerate(param_grid):
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=mss, random_state=42)
    clf.fit(X_train.toarray(), y_train)  # dataset is small enough to densify

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev))

    experiment_data["min_samples_split"]["SPR_BENCH"]["metrics"]["train"].append(
        train_acc
    )
    experiment_data["min_samples_split"]["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["min_samples_split"]["SPR_BENCH"]["losses"]["train"].append(
        1 - train_acc
    )
    experiment_data["min_samples_split"]["SPR_BENCH"]["losses"]["val"].append(
        1 - val_acc
    )

    node_count = clf.tree_.node_count
    if val_acc > best_val_acc or (
        np.isclose(val_acc, best_val_acc) and node_count < best_nodes
    ):
        best_clf, best_val_acc, best_nodes = clf, val_acc, node_count
        experiment_data["min_samples_split"]["SPR_BENCH"]["best_param"] = mss

    print(
        f"min_samples_split={mss:>2} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
    )

# ---------- Final evaluation ----------
test_acc = accuracy_score(y_test, best_clf.predict(X_test))
sefa_score = compute_sefa(best_clf, X_test.toarray(), y_test)
experiment_data["min_samples_split"]["SPR_BENCH"]["predictions"] = best_clf.predict(
    X_test
).tolist()
experiment_data["min_samples_split"]["SPR_BENCH"]["test_accuracy"] = test_acc
experiment_data["min_samples_split"]["SPR_BENCH"]["sefa"] = sefa_score

print(
    f"Best min_samples_split: {experiment_data['min_samples_split']['SPR_BENCH']['best_param']}"
)
print(f"Test accuracy: {test_acc:.4f}")
print(f"SEFA: {sefa_score:.4f}")

# ---------- Save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir, 'experiment_data.npy')}")
