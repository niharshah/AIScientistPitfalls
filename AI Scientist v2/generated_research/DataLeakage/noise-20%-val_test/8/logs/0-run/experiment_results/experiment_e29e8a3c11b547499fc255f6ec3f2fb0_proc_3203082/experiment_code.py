import os, pathlib, random, string, numpy as np, torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict

# ---------------- Saving dict --------------------
experiment_data = {
    "criterion_tuning": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "best_criterion": None,
            "per_criterion": {},
        }
    }
}

# ---------------- Device -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Data utils ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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
                string.ascii_uppercase + string.digits, k=random.randint(6, 12)
            )
        )

    def label_fn(s):
        return 1 if s.count("A") > s.count("B") else 0

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

# ---------------- Vectoriser ---------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
X_train = vectorizer.fit_transform(data["train"]["sequence"]).toarray()
X_dev = vectorizer.transform(data["dev"]["sequence"]).toarray()
X_test = vectorizer.transform(data["test"]["sequence"]).toarray()
y_train = np.array(data["train"]["label"])
y_dev = np.array(data["dev"]["label"])
y_test = np.array(data["test"]["label"])


# ---------------- Helper funcs -------------------
def decision_tree_single_pred(clf: DecisionTreeClassifier, x_row: np.ndarray) -> int:
    t = clf.tree_
    node = 0
    while t.children_left[node] != _tree.TREE_LEAF:
        node = (
            t.children_left[node]
            if x_row[t.feature[node]] <= t.threshold[node]
            else t.children_right[node]
        )
    return np.argmax(t.value[node][0])


def compute_sefa(clf, X, y):
    preds = clf.predict(X)
    return np.mean(
        [
            decision_tree_single_pred(clf, X[i]) == preds[i] == y[i]
            for i in range(len(y))
        ]
    )


# ---------------- Hyper-parameter search ----------
criteria = ["gini", "entropy", "log_loss"]
best_val_acc = -1.0
best_clf = None
for crit in criteria:
    try:
        clf = DecisionTreeClassifier(max_depth=5, random_state=42, criterion=crit)
        clf.fit(X_train, y_train)
    except ValueError as e:
        print(f"Criterion '{crit}' not supported in this sklearn version -> skipped.")
        continue

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev))
    train_loss, val_loss = 1 - train_acc, 1 - val_acc

    experiment_data["criterion_tuning"]["SPR_BENCH"]["per_criterion"][crit] = {
        "metrics": {"train": [train_acc], "val": [val_acc]},
        "losses": {"train": [train_loss], "val": [val_loss]},
    }
    print(f"Criterion={crit}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc, best_clf = val_acc, clf
        experiment_data["criterion_tuning"]["SPR_BENCH"]["best_criterion"] = crit
        experiment_data["criterion_tuning"]["SPR_BENCH"]["metrics"]["train"] = [
            train_acc
        ]
        experiment_data["criterion_tuning"]["SPR_BENCH"]["metrics"]["val"] = [val_acc]
        experiment_data["criterion_tuning"]["SPR_BENCH"]["losses"]["train"] = [
            train_loss
        ]
        experiment_data["criterion_tuning"]["SPR_BENCH"]["losses"]["val"] = [val_loss]

print(
    f"Best criterion: {experiment_data['criterion_tuning']['SPR_BENCH']['best_criterion']} (val_acc={best_val_acc:.4f})"
)

# ---------------- Test evaluation ----------------
test_acc = accuracy_score(y_test, best_clf.predict(X_test))
sefa = compute_sefa(best_clf, X_test, y_test)
experiment_data["criterion_tuning"]["SPR_BENCH"]["predictions"] = best_clf.predict(
    X_test
).tolist()
experiment_data["criterion_tuning"]["SPR_BENCH"]["ground_truth"] = y_test.tolist()
print(f"Test accuracy: {test_acc:.4f}")
print(f"Self-Explain Fidelity Accuracy (SEFA): {sefa:.4f}")

# ---------------- Save artefacts -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")
