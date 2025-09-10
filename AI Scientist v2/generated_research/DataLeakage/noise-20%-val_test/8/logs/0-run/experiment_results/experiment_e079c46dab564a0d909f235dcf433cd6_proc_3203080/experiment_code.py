import os, pathlib, random, string, numpy as np, torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from datasets import DatasetDict, Dataset
from typing import Dict

# ---------- Working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Device info ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- Data utils ----------
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


# ---------- Load data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
data = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else create_synthetic_spr()
print({k: len(v) for k, v in data.items()})

# ---------- Vectorization ----------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
X_train = vectorizer.fit_transform(data["train"]["sequence"]).toarray()
X_dev = vectorizer.transform(data["dev"]["sequence"]).toarray()
X_test = vectorizer.transform(data["test"]["sequence"]).toarray()
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


def compute_sefa(clf, X, y):
    preds = clf.predict(X)
    successes = sum(
        (decision_tree_single_pred(clf, X[i]) == preds[i] == y[i])
        for i in range(len(y))
    )
    return successes / len(y)


# ---------- Hyperparameter tuning ----------
candidates = [None, "sqrt", "log2", 0.2, 0.5]
best_val_acc, best_model, best_hp = -1, None, None

experiment_data = {
    "max_features_tuning": {
        "SPR_BENCH": {
            "hyperparams": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for hp in candidates:
    clf = DecisionTreeClassifier(max_depth=5, random_state=42, max_features=hp)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev))
    experiment_data["max_features_tuning"]["SPR_BENCH"]["hyperparams"].append(hp)
    experiment_data["max_features_tuning"]["SPR_BENCH"]["metrics"]["train"].append(
        train_acc
    )
    experiment_data["max_features_tuning"]["SPR_BENCH"]["metrics"]["val"].append(
        val_acc
    )
    experiment_data["max_features_tuning"]["SPR_BENCH"]["losses"]["train"].append(
        1 - train_acc
    )
    experiment_data["max_features_tuning"]["SPR_BENCH"]["losses"]["val"].append(
        1 - val_acc
    )
    print(f"HP={hp}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc, best_model, best_hp = val_acc, clf, hp

print(f"Best max_features setting: {best_hp} (val_acc={best_val_acc:.4f})")

# ---------- Final evaluation ----------
test_acc = accuracy_score(y_test, best_model.predict(X_test))
sefa = compute_sefa(best_model, X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"SEFA: {sefa:.4f}")

experiment_data["max_features_tuning"]["SPR_BENCH"]["predictions"] = best_model.predict(
    X_test
).tolist()
experiment_data["max_features_tuning"]["SPR_BENCH"]["ground_truth"] = y_test.tolist()
experiment_data["max_features_tuning"]["SPR_BENCH"]["test_accuracy"] = test_acc
experiment_data["max_features_tuning"]["SPR_BENCH"]["sefa"] = sefa
experiment_data["max_features_tuning"]["SPR_BENCH"]["best_max_features"] = best_hp

# ---------- Save artefacts ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")
