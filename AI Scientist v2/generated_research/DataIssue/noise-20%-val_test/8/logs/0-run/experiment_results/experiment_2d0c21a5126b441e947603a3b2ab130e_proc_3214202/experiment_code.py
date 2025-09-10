import os, pathlib, json, numpy as np, matplotlib.pyplot as plt, torch
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# ----------------- house-keeping ----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------- data -------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def get_dataset() -> DatasetDict:
    maybe = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        ds = load_spr_bench(maybe)
        print("Loaded real SPR_BENCH at", maybe)
        return ds
    except Exception:
        print("Generating synthetic toy data.")
        rng = np.random.default_rng(0)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                size = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=size))
                labels.append(int(seq.count("A") % 2 == 0))
                seqs.append(seq)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


dsets = get_dataset()
dataset_name = "SPR_BENCH"


# ----------------- helper to build vocab / vectors ---------------
def build_char_stats(split):
    freq = {}
    for s in dsets[split]["sequence"]:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
    return freq


train_freq = build_char_stats("train")
all_chars_sorted = sorted(train_freq.items(), key=lambda kv: kv[1])  # asc by freq
print("Char frequencies:", train_freq)


def train_eval(drop_chars=None):
    drop_chars = set(drop_chars or [])
    chars = sorted([c for c in train_freq if c not in drop_chars])
    char2idx = {c: i for i, c in enumerate(chars)}
    V = len(chars)

    def seq_to_vec(seq: str) -> np.ndarray:
        v = np.zeros(V, dtype=np.float32)
        for ch in seq:
            if ch in char2idx:
                v[char2idx[ch]] += 1.0
        return v

    def vec_split(split):
        X = np.stack([seq_to_vec(s) for s in dsets[split]["sequence"]])
        y = np.array(dsets[split]["label"])
        return X, y

    X_tr, y_tr = vec_split("train")
    X_val, y_val = vec_split("dev")
    X_te, y_te = vec_split("test")

    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_tr, y_tr)

    # metrics
    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    val_proba = clf.predict_proba(X_val)
    val_loss = log_loss(y_val, val_proba)
    val_acc = accuracy_score(y_val, np.argmax(val_proba, 1))
    test_pred = clf.predict(X_te)
    test_acc = accuracy_score(y_te, test_pred)

    result = {
        "metrics": {"train": [train_acc], "val": [val_acc], "test": [test_acc]},
        "losses": {"train": [], "val": [val_loss]},
        "predictions": test_pred.tolist(),
        "ground_truth": y_te.tolist(),
    }

    return result, clf, chars


# ----------------- run baseline -----------------------------------
experiment_data = {}
baseline_key = "baseline_full_vocab"
baseline_res, baseline_clf, baseline_chars = train_eval()
experiment_data[baseline_key] = {dataset_name: baseline_res}
print("Baseline accuracy:", baseline_res["metrics"]["test"][0])

# save baseline confusion matrix and rules
cm = confusion_matrix(baseline_res["ground_truth"], baseline_res["predictions"])
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Baseline Confusion Matrix")
plt.colorbar(im, ax=ax)
plt.savefig(os.path.join(working_dir, "confusion_matrix.png"))
plt.close()


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
            rec(tree_.children_left[node], cur + [f"{name}<={thr:.1f}"])
            rec(tree_.children_right[node], cur + [f"{name}>{thr:.1f}"])
        else:
            pred = np.argmax(tree_.value[node][0])
            rule = " AND ".join(cur) if cur else "TRUE"
            rules.append(f"IF {rule} THEN label={pred}")

    rec(0, [])
    return rules


rules = tree_to_rules(baseline_clf, baseline_chars)
with open(os.path.join(working_dir, "extracted_rules.txt"), "w") as f:
    f.write("\n".join(rules))

# --------------- Character-vocabulary reduction ablations ----------
low_freq_sorted = [c for c, _ in all_chars_sorted]
ablations_to_run = []
if low_freq_sorted:
    ablations_to_run.append([low_freq_sorted[0]])  # drop 1 rare char
if len(low_freq_sorted) >= 2:
    ablations_to_run.append(low_freq_sorted[:2])  # drop 2 rare chars

for drop in ablations_to_run:
    key = f"vocab_reduction_drop_{''.join(drop)}"
    res, _, _ = train_eval(drop_chars=drop)
    experiment_data[key] = {dataset_name: res}
    print(f"{key} -> test acc: {res['metrics']['test'][0]:.4f}")

# ---------------- save everything ---------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy with keys:", list(experiment_data.keys()))
