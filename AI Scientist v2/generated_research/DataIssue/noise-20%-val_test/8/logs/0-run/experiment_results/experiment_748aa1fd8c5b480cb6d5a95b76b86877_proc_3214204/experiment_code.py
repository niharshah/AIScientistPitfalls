import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch, warnings
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- WORKING DIR ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


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
        return spr, "SPR_BENCH"
    except Exception:
        print("Could not load real SPR_BENCH; generating synthetic toy dataset.")
        rng = np.random.default_rng(42)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for _ in range(n):
                length = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=length))
                label = int(seq.count("A") % 2 == 0)  # parity on 'A'
                seqs.append(seq)
                labels.append(label)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200)), "synthetic_toy"


dsets, dataset_name = get_dataset()

# ---------------------- PREPROCESSING -------------------------------
chars = sorted(
    {ch for split in dsets for seq in dsets[split]["sequence"] for ch in seq}
)
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


# ---------------------- RULE EXTRACTION UTIL ------------------------
def tree_to_rules(clf, feature_names):
    tree_ = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    def recurse(node, cur_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, thr = feature_name[node], tree_.threshold[node]
            recurse(tree_.children_left[node], cur_rule + [f"{name} <= {thr:.1f}"])
            recurse(tree_.children_right[node], cur_rule + [f"{name} > {thr:.1f}"])
        else:
            rule = " AND ".join(cur_rule) if cur_rule else "TRUE"
            pred = np.argmax(tree_.value[node][0])
            paths.append(f"IF {rule} THEN label={pred}")

    recurse(0, [])
    return paths


# ---------------------- ABLATION LOOP -------------------------------
depth_settings = [1, 3, 5, 10, None]  # None == unlimited depth
experiment_data = {
    "tree_depth_sensitivity": {
        dataset_name: {
            "depths": [],
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": []},
            "rule_counts": [],
            "predictions": {},
            "ground_truth": y_test.tolist(),
        }
    }
}

for depth in depth_settings:
    print(f"\n--- Training tree with max_depth={depth} ---")
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)

    # Predictions & losses
    train_proba = clf.predict_proba(X_train)
    dev_proba = clf.predict_proba(X_dev)
    test_pred = clf.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_dev, clf.predict(X_dev))
    test_acc = accuracy_score(y_test, test_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_loss = log_loss(y_train, train_proba)
        val_loss = log_loss(y_dev, dev_proba)

    # Rule extraction
    rules = tree_to_rules(clf, chars)
    rule_file = os.path.join(working_dir, f"rules_depth_{depth}.txt")
    with open(rule_file, "w") as f:
        f.write("\n".join(rules))
    rule_count = len(rules)
    print(
        f"Depth {depth}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
        f"test_acc={test_acc:.3f}, rules={rule_count}"
    )

    # Save confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (depth={depth})")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    cm_path = os.path.join(working_dir, f"confusion_matrix_depth_{depth}.png")
    plt.savefig(cm_path)
    plt.close()

    # Store in experiment data
    ed = experiment_data["tree_depth_sensitivity"][dataset_name]
    ed["depths"].append("None" if depth is None else depth)
    ed["metrics"]["train"].append(train_acc)
    ed["metrics"]["val"].append(val_acc)
    ed["metrics"]["test"].append(test_acc)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["rule_counts"].append(rule_count)
    ed["predictions"][str(depth)] = test_pred.tolist()

# ---------------------- SAVE EXPERIMENT DATA ------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy with keys:", list(experiment_data.keys()))
