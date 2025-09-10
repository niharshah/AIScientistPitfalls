import os, pathlib, random, string, numpy as np, torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict

# ----------------- House-keeping -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- Data utils -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def create_synthetic_spr(n_train=2000, n_dev=500, n_test=500) -> DatasetDict:
    def gen_seq():
        return "".join(
            random.choices(
                list(string.ascii_uppercase) + list(string.digits),
                k=random.randint(6, 12),
            )
        )

    def label_fn(s):
        return 1 if s.count("A") > s.count("B") else 0

    def build(n):
        seqs = [gen_seq() for _ in range(n)]
        labels = [label_fn(s) for s in seqs]
        ids = list(range(n))
        return Dataset.from_dict({"id": ids, "sequence": seqs, "label": labels})

    return DatasetDict(train=build(n_train), dev=build(n_dev), test=build(n_test))


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
data = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else create_synthetic_spr()
print({k: len(v) for k, v in data.items()})

# ----------------- Vectorisation ---------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
X_train = vectorizer.fit_transform(data["train"]["sequence"])
y_train = np.array(data["train"]["label"])
X_dev = vectorizer.transform(data["dev"]["sequence"])
y_dev = np.array(data["dev"]["label"])
X_test = vectorizer.transform(data["test"]["sequence"])
y_test = np.array(data["test"]["label"])


# ----------------- SEFA helper -----------------
def decision_tree_single_pred(clf: DecisionTreeClassifier, x_row: np.ndarray) -> int:
    tree = clf.tree_
    node = 0
    while tree.children_left[node] != _tree.TREE_LEAF:
        node = (
            tree.children_left[node]
            if x_row[tree.feature[node]] <= tree.threshold[node]
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


# --------------- Hyper-parameter tuning ----------
splitter_options = ["best", "random"]
records = {
    "train_acc": [],
    "val_acc": [],
    "train_loss": [],
    "val_loss": [],
    "splitter": [],
    "model": [],
}

for split_opt in splitter_options:
    clf = DecisionTreeClassifier(max_depth=5, random_state=42, splitter=split_opt)
    clf.fit(X_train.toarray(), y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev))
    records["splitter"].append(split_opt)
    records["train_acc"].append(train_acc)
    records["val_acc"].append(val_acc)
    records["train_loss"].append(1 - train_acc)
    records["val_loss"].append(1 - val_acc)
    records["model"].append(clf)
    print(f"Splitter={split_opt}: val_acc={val_acc:.4f}")

best_idx = int(np.argmax(records["val_acc"]))
best_splitter = records["splitter"][best_idx]
best_clf = records["model"][best_idx]
print(f"Best splitter: {best_splitter} (val_acc={records['val_acc'][best_idx]:.4f})")

# --------------- Test evaluation ----------------
test_acc = accuracy_score(y_test, best_clf.predict(X_test))
sefa = compute_sefa(best_clf, X_test.toarray(), y_test)
print(f"Test accuracy (best model): {test_acc:.4f}")
print(f"SEFA (best model): {sefa:.4f}")

# --------------- Experiment data dict -----------
experiment_data = {
    "splitter_tuning": {
        "SPR_BENCH": {
            "metrics": {"train": records["train_acc"], "val": records["val_acc"]},
            "losses": {"train": records["train_loss"], "val": records["val_loss"]},
            "chosen_splitter": best_splitter,
            "test_accuracy": test_acc,
            "sefa": sefa,
            "predictions": best_clf.predict(X_test).tolist(),
            "ground_truth": y_test.tolist(),
            "splitter_options": splitter_options,
        }
    }
}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir, "experiment_data.npy")}')
