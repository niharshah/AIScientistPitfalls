import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# ------------------------- HOUSEKEEPING -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment data container (required naming convention)
experiment_data = {}


# ------------------------- DATA LOADING -----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


def get_dataset() -> DatasetDict:
    possible = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        ds = load_spr_bench(possible)
        print("Loaded real SPR_BENCH.")
        return ds
    except Exception:
        print("Could not load real SPR_BENCH; generating tiny synthetic one.")
        rng = np.random.default_rng(42)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for _ in range(n):
                length = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=length))
                labels.append(int(seq.count("A") % 2 == 0))  # parity of A
                seqs.append(seq)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


dsets = get_dataset()

# ------------------------- VOCAB & VECTORS --------------------------
chars = sorted({ch for split in dsets for s in dsets[split]["sequence"] for ch in s})
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)
print("Character vocab:", chars)


def vec_count(seq: str) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    for ch in seq:
        if ch in char2idx:
            v[char2idx[ch]] += 1.0
    return v


def vec_presence(seq: str) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    for ch in set(seq):
        if ch in char2idx:
            v[char2idx[ch]] = 1.0
    return v


def vectorise_split(split, mode: str):
    if mode == "frequency":
        f = vec_count
    else:
        f = vec_presence
    X = np.stack([f(s) for s in dsets[split]["sequence"]])
    y = np.array(dsets[split]["label"])
    return X, y


# ------------------------- RULE UTILS -------------------------------
def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, cur):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            nm = feature_name[node]
            thr = tree_.threshold[node]
            recurse(tree_.children_left[node], cur + [f"{nm} <= {thr:.1f}"])
            recurse(tree_.children_right[node], cur + [f"{nm} > {thr:.1f}"])
        else:
            proba = tree_.value[node][0]
            pred = np.argmax(proba)
            cond = " AND ".join(cur) if cur else "TRUE"
            rules.append(f"IF {cond} THEN label={pred}")

    recurse(0, [])
    return rules


def avg_rule_len(rule_list):
    lengths = []
    for r in rule_list:
        body = r.split(" THEN")[0].replace("IF ", "")
        lengths.append(1 if body == "TRUE" else body.count(" AND ") + 1)
    return float(np.mean(lengths)) if lengths else 0.0


# ------------------------- EXPERIMENT LOOP --------------------------
variants = {
    "baseline_frequency": "frequency",
    "presence_ablation": "presence",
}
for tag, mode in variants.items():
    print("\n=== Running variant:", tag, "===")
    # vectorise
    X_train, y_train = vectorise_split("train", mode)
    X_dev, y_dev = vectorise_split("dev", mode)
    X_test, y_test = vectorise_split("test", mode)

    # model
    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)

    # rules + statistics
    rules = tree_to_rules(clf, chars)
    depth = clf.get_depth()
    avg_len = avg_rule_len(rules)
    rules_path = os.path.join(working_dir, f"{tag}_rules.txt")
    with open(rules_path, "w") as f:
        f.write("\n".join(rules))
    print(f"Saved {len(rules)} rules to", rules_path)

    # metrics
    dev_proba = clf.predict_proba(X_dev)
    val_loss = log_loss(y_dev, dev_proba)
    dev_acc = accuracy_score(y_dev, clf.predict(X_dev))
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({tag})")
    plt.colorbar(im, ax=ax)
    cm_path = os.path.join(working_dir, f"{tag}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print("Saved confusion matrix to", cm_path)

    # store results
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {
                "train": [],
                "val": [dev_acc],
                "test": [test_acc],
                "rule_depth": [depth],
                "avg_rule_len": [avg_len],
            },
            "losses": {
                "train": [],
                "val": [val_loss],
            },
            "predictions": test_pred.tolist(),
            "ground_truth": y_test.tolist(),
        }
    }
    print(
        f"{tag}: val_loss={val_loss:.4f}, val_acc={dev_acc:.4f}, test_acc={test_acc:.4f}, depth={depth}, avg_rule_len={avg_len:.2f}"
    )

# ------------------------- SAVE EXPERIMENT DATA ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "\nSaved all experiment data to", os.path.join(working_dir, "experiment_data.npy")
)
