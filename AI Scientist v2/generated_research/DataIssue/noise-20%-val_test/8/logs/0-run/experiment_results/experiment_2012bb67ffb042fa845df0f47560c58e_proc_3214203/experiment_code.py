# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch, random
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# --------------------------------------------------------------------
# GPU / CPU handling (mandatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------------- DATA LOADING --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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


def get_dataset() -> DatasetDict:
    root = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        d = load_spr_bench(root)
        print("Loaded real SPR_BENCH from", root)
        return d
    except Exception:
        print("Could not load real SPR_BENCH; generating synthetic toy dataset.")
        rng = np.random.default_rng(42)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                length = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=length))
                label = int(seq.count("A") % 2 == 0)  # parity on 'A'
                seqs.append(seq)
                labels.append(label)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


dsets = get_dataset()
dataset_name = "SPR_BENCH"

# ---------------------- PREPROCESSING -------------------------------
chars = sorted(
    {ch for split in dsets for seq in dsets[split]["sequence"] for ch in seq}
)
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)
print("Character vocab:", chars)


def seq_to_vec(s):
    v = np.zeros(V, dtype=np.float32)
    for ch in s:
        if ch in char2idx:
            v[char2idx[ch]] += 1.0
    return v


def vectorise(split):
    X = np.stack([seq_to_vec(s) for s in dsets[split]["sequence"]])
    y = np.array(dsets[split]["label"])
    return X, y


X_train_base, y_train_base = vectorise("train")
X_dev, y_dev = vectorise("dev")
X_test, y_test = vectorise("test")


# ---------------------- RULE EXTRACTION -----------------------------
def path_to_rule(tree, feature_names):
    tree_ = tree.tree_
    feat_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undef"
        for i in tree_.feature
    ]
    rules = []

    def rec(node, curr):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feat_name[node]
            thr = tree_.threshold[node]
            rec(tree_.children_left[node], curr + [f"{name}<={thr:.1f}"])
            rec(tree_.children_right[node], curr + [f"{name}>{thr:.1f}"])
        else:
            pred = np.argmax(tree_.value[node][0])
            cond = " AND ".join(curr) if curr else "TRUE"
            rules.append(f"IF {cond} THEN label={pred}")

    rec(0, [])
    return rules


# ---------------------- LABEL NOISE ABLATION ------------------------
noise_levels = [0.0, 0.1, 0.2, 0.3]
experiment_data = {"label_noise_robustness": {}}
rng_global = np.random.default_rng(0)

for pct in noise_levels:
    noise_key = f"noise_{int(pct*100)}"
    ds_key = f"{dataset_name}_{noise_key}"
    print(f"\n=== Training with {pct*100:.0f}% noisy labels ===")
    y_train = y_train_base.copy()
    if pct > 0:
        n_flip = int(len(y_train) * pct)
        idx_to_flip = rng_global.choice(len(y_train), size=n_flip, replace=False)
        unique_labels = np.unique(y_train)
        for idx in idx_to_flip:
            orig = y_train[idx]
            choices = unique_labels[unique_labels != orig]
            y_train[idx] = rng_global.choice(choices)
    # Train model
    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_train_base, y_train)
    # Extract rules
    rules = path_to_rule(clf, chars)
    with open(os.path.join(working_dir, f"rules_{noise_key}.txt"), "w") as f:
        f.write("\n".join(rules))
    # Metrics
    train_pred = clf.predict(X_train_base)
    dev_pred = clf.predict(X_dev)
    test_pred = clf.predict(X_test)
    dev_proba = clf.predict_proba(X_dev)
    val_loss = log_loss(y_dev, dev_proba)
    train_acc = accuracy_score(y_train, train_pred)
    dev_acc = accuracy_score(y_dev, dev_pred)
    test_acc = accuracy_score(y_test, test_pred)
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion {noise_key}")
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(working_dir, f"cm_{noise_key}.png"))
    plt.close()
    # Store
    experiment_data["label_noise_robustness"][ds_key] = {
        "metrics": {"train": [train_acc], "val": [dev_acc], "test": [test_acc]},
        "losses": {"train": [], "val": [val_loss]},
        "predictions": test_pred.tolist(),
        "ground_truth": y_test.tolist(),
        "noise_level": pct,
    }

# ---------------------- SAVE EXPERIMENT DATA ------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "\nSaved experiment_data.npy with keys:",
    list(experiment_data["label_noise_robustness"].keys()),
)
