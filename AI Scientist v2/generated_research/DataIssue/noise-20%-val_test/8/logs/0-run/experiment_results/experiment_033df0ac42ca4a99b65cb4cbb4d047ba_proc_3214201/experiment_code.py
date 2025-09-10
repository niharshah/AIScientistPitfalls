import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

# --------------------------------------------------------------------
# GPU / CPU handling (kept for completeness)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------------------------------------------------------

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------
# Synthetic dataset generators
rng_global = np.random.default_rng(42)
VOCAB = list("ABC")


def _gen_split(n, rule_fn, seed):
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        length = rng.integers(4, 8)
        seq = "".join(rng.choice(VOCAB, size=length))
        labels.append(rule_fn(seq))
        seqs.append(seq)
    return seqs, labels


def build_dataset(rule_fn, base_seed):
    # returns dict with keys train/dev/test -> {'sequence': [...], 'label': [...]}
    sizes = {"train": 600, "dev": 200, "test": 200}
    dset = {}
    for i, split in enumerate(sizes):
        seqs, labels = _gen_split(sizes[split], rule_fn, base_seed + i)
        dset[split] = {"sequence": seqs, "label": labels}
    return dset


# Rules
rule_parity_A = lambda s: int(s.count("A") % 2 == 0)
rule_majority_B = lambda s: int(s.count("B") > len(s) / 2)
rule_last_C = lambda s: int(s[-1] == "C")

DATASETS_INFO = {
    "parity_A": (rule_parity_A, 100),
    "majority_B": (rule_majority_B, 200),
    "last_C": (rule_last_C, 300),
}

# --------------------------------------------------------------------
# Vectoriser (bag of chars)
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)


def seq_to_vec(seq):
    v = np.zeros(V, dtype=np.float32)
    for ch in seq:
        v[CHAR2IDX[ch]] += 1.0
    return v


def vectorise_split(dsplit):
    X = np.stack([seq_to_vec(s) for s in dsplit["sequence"]])
    y = np.array(dsplit["label"])
    return X, y


# --------------------------------------------------------------------
# Rule extraction helper
def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feat_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def rec(node, cur):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feat_name[node]
            thr = tree_.threshold[node]
            rec(tree_.children_left[node], cur + [f"{name} <= {thr:.1f}"])
            rec(tree_.children_right[node], cur + [f"{name} > {thr:.1f}"])
        else:
            pred = np.argmax(tree_.value[node][0])
            rule = " AND ".join(cur) if cur else "TRUE"
            rules.append(f"IF {rule} THEN label={pred}")

    rec(0, [])
    return rules


# --------------------------------------------------------------------
# Experiment loop
experiment_data = {"multi_synth_generalization": {}}

for dname, (rule_fn, seed) in DATASETS_INFO.items():
    print(f"\n=== Processing dataset: {dname} ===")
    dset = build_dataset(rule_fn, seed)

    # Vectorise
    X_train, y_train = vectorise_split(dset["train"])
    X_dev, y_dev = vectorise_split(dset["dev"])
    X_test, y_test = vectorise_split(dset["test"])

    # Model
    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)

    # Metrics
    train_pred = clf.predict(X_train)
    dev_proba = clf.predict_proba(X_dev)
    test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    dev_acc = accuracy_score(y_dev, clf.predict(X_dev))
    val_loss = log_loss(y_dev, dev_proba, labels=[0, 1])
    test_acc = accuracy_score(y_test, test_pred)
    sefa = test_acc  # identical for this setting
    complexity = clf.tree_.node_count

    # Save rules
    rules = tree_to_rules(clf, VOCAB)
    with open(os.path.join(working_dir, f"extracted_rules_{dname}.txt"), "w") as f:
        f.write("\n".join(rules))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{dname} Confusion")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(working_dir, f"confusion_matrix_{dname}.png"))
    plt.close()

    # Log experiment data
    experiment_data["multi_synth_generalization"][dname] = {
        "metrics": {
            "train": [train_acc],
            "val": [dev_acc],
            "test": [test_acc],
            "sefa": [sefa],
        },
        "losses": {"train": [], "val": [val_loss]},
        "rule_complexity": complexity,
        "predictions": test_pred.tolist(),
        "ground_truth": y_test.tolist(),
    }

# --------------------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "\nSaved experiment_data.npy with keys:",
    list(experiment_data["multi_synth_generalization"].keys()),
)
