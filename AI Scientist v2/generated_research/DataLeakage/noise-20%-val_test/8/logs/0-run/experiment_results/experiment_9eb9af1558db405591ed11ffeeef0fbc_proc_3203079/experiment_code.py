import os, pathlib, random, string, numpy as np, torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from datasets import Dataset, DatasetDict

# ------------------- Setup & data dict -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "ccp_alpha_tuning": {
        "SPR_BENCH": {
            "tested_alphas": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- Data loading ------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


def create_synthetic_spr(n_train=2000, n_dev=500, n_test=500) -> DatasetDict:
    def gen_seq():
        return "".join(
            random.choices(
                list(string.ascii_uppercase) + list(string.digits),
                k=random.randint(6, 12),
            )
        )

    def label_fn(s):  # rule
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

# ------------------- Vectorization -----------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
X_train = vectorizer.fit_transform(data["train"]["sequence"])
X_dev = vectorizer.transform(data["dev"]["sequence"])
X_test = vectorizer.transform(data["test"]["sequence"])
y_train = np.array(data["train"]["label"])
y_dev = np.array(data["dev"]["label"])
y_test = np.array(data["test"]["label"])


# ------------------- SEFA helper -------------------------
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
    ok = 0
    for i in range(X.shape[0]):
        if preds[i] == y[i] == decision_tree_single_pred(clf, X[i]):
            ok += 1
    return ok / X.shape[0]


# ------------------- Hyper-parameter tuning --------------
alphas = [0.0, 1e-4, 1e-3, 1e-2]
best_model, best_alpha, best_val_acc = None, None, -1.0

for alpha in alphas:
    clf = DecisionTreeClassifier(max_depth=5, random_state=42, ccp_alpha=alpha)
    clf.fit(X_train.toarray(), y_train)  # small enough to densify

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev))

    experiment_data["ccp_alpha_tuning"]["SPR_BENCH"]["tested_alphas"].append(alpha)
    experiment_data["ccp_alpha_tuning"]["SPR_BENCH"]["metrics"]["train"].append(
        train_acc
    )
    experiment_data["ccp_alpha_tuning"]["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["ccp_alpha_tuning"]["SPR_BENCH"]["losses"]["train"].append(
        1 - train_acc
    )
    experiment_data["ccp_alpha_tuning"]["SPR_BENCH"]["losses"]["val"].append(
        1 - val_acc
    )

    print(f"Alpha={alpha:.4g}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc, best_alpha, best_model = val_acc, alpha, clf

print(f"Best alpha by dev accuracy: {best_alpha} (val_acc={best_val_acc:.4f})")

# ------------------- Test evaluation ---------------------
test_acc = accuracy_score(y_test, best_model.predict(X_test))
sefa = compute_sefa(best_model, X_test.toarray(), y_test)
experiment_data["ccp_alpha_tuning"]["SPR_BENCH"]["predictions"] = best_model.predict(
    X_test
).tolist()
experiment_data["ccp_alpha_tuning"]["SPR_BENCH"]["ground_truth"] = y_test.tolist()
print(f"Test accuracy: {test_acc:.4f}")
print(f"SEFA: {sefa:.4f}")

# ------------------- Save artefacts ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")
