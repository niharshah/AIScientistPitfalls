import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# -------------------- ENV / IO --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
work = pathlib.Path(os.getcwd()) / "working"
work.mkdir(exist_ok=True)


# -------------------- DATA ------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def synthetic_toy() -> DatasetDict:
    rng, vocab = np.random.default_rng(42), list("ABC")

    def gen(n):
        seqs, labels = [], []
        for i in range(n):
            l = rng.integers(4, 8)
            s = "".join(rng.choice(vocab, l))
            labels.append(int(s.count("A") % 2 == 0))
            seqs.append(s)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


try:
    dsets = load_spr_bench(pathlib.Path(os.getcwd()) / "SPR_BENCH")
    print("Loaded real SPR_BENCH.")
except Exception:
    print("Falling back to synthetic data.")
    dsets = synthetic_toy()

# build vocab
chars = sorted({c for split in dsets for s in dsets[split]["sequence"] for c in s})
V = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
print("Vocabulary:", chars)


# -------------- FEATURE CONSTRUCTION -------------
def vec_bag(seq: str) -> np.ndarray:
    v = np.zeros(V, np.float32)
    for ch in seq:
        if ch in char2idx:
            v[char2idx[ch]] += 1
    return v


def vec_positional(seq: str) -> np.ndarray:
    bag = vec_bag(seq)
    first = np.zeros(V, np.float32)
    last = np.zeros(V, np.float32)
    if seq:
        first[char2idx.get(seq[0], 0)] = 1
        last[char2idx.get(seq[-1], 0)] = 1
    return np.concatenate([bag, first, last])


def vectorise(split, fn):
    X = np.stack([fn(s) for s in dsets[split]["sequence"]])
    y = np.array(dsets[split]["label"])
    return X, y


# -------------- MODEL / UTILS --------------------
def train_and_eval(name, vec_fn, feature_names):
    X_tr, y_tr = vectorise("train", vec_fn)
    X_dv, y_dv = vectorise("dev", vec_fn)
    X_te, y_te = vectorise("test", vec_fn)

    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_tr, y_tr)

    dev_proba = clf.predict_proba(X_dv)
    val_loss = log_loss(y_dv, dev_proba)
    val_acc = accuracy_score(y_dv, clf.predict(X_dv))
    test_pred = clf.predict(X_te)
    test_acc = accuracy_score(y_te, test_pred)

    # Rules
    def tree_to_rules(tree, f_names):
        tree_, rules = tree.tree_, []
        fn = [
            f_names[i] if i != _tree.TREE_UNDEFINED else "undef" for i in tree_.feature
        ]

        def rec(node, cur):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                th = tree_.threshold[node]
                rec(tree_.children_left[node], cur + [f"{fn[node]}<={th:.1f}"])
                rec(tree_.children_right[node], cur + [f"{fn[node]}>{th:.1f}"])
            else:
                pred = np.argmax(tree_.value[node][0])
                rules.append(
                    "IF "
                    + (" AND ".join(cur) if cur else "TRUE")
                    + f" THEN label={pred}"
                )

        rec(0, [])
        return rules

    rules = tree_to_rules(clf, feature_names)
    with open(work / f"rules_{name}.txt", "w") as f:
        f.write("\n".join(rules))
    print(f"{name}: wrote {len(rules)} rules.")

    # Confusion matrix plot
    cm = confusion_matrix(y_te, test_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(title=f"{name} Confusion", xlabel="Pred", ylabel="True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(work / f"cm_{name}.png")
    plt.close()

    return {
        "metrics": {"train": [], "val": [val_acc], "test": [test_acc]},
        "losses": {"train": [], "val": [val_loss]},
        "predictions": test_pred.tolist(),
        "ground_truth": y_te.tolist(),
    }


# feature names lists
bag_names = [f"cnt_{c}" for c in chars]
pos_names = bag_names + [f"first_{c}" for c in chars] + [f"last_{c}" for c in chars]

# ---------------- RUN EXPERIMENTS ----------------
experiment_data = {
    "bag_of_chars": {"SPR_BENCH": None},
    "positional": {"SPR_BENCH": None},
}

experiment_data["bag_of_chars"]["SPR_BENCH"] = train_and_eval("bag", vec_bag, bag_names)
experiment_data["positional"]["SPR_BENCH"] = train_and_eval(
    "positional", vec_positional, pos_names
)

np.save(work / "experiment_data.npy", experiment_data)
print("Saved experiment_data.npy to", work)
