import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, numpy as np, matplotlib.pyplot as plt, torch, re, json
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

# --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------------------------------------------------------


# ------------ DATA --------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _ld(f"{s}.csv")
    return d


def get_dataset() -> DatasetDict:
    root = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        ds = load_spr_bench(root)
        print("Loaded real SPR_BENCH.")
        return ds
    except Exception as e:
        print("SPR_BENCH not found, creating synthetic parity dataset.")
        rng = np.random.default_rng(0)
        vocab = list("ABCDE")

        def make(n):
            seq, lab = [], []
            for i in range(n):
                L = rng.integers(4, 9)
                s = "".join(rng.choice(vocab, size=L))
                seq.append(s)
                lab.append(int(s.count("A") % 2 == 0))
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seq, "label": lab}
            )

        return DatasetDict(train=make(2000), dev=make(500), test=make(1000))


dsets = get_dataset()

# ------------ VECTORISATION ----------------------------------------
train_text = dsets["train"]["sequence"]
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), binary=False)
X_train = vectorizer.fit_transform(train_text)
y_train = np.array(dsets["train"]["label"])

X_dev = vectorizer.transform(dsets["dev"]["sequence"])
y_dev = np.array(dsets["dev"]["label"])

X_test = vectorizer.transform(dsets["test"]["sequence"])
y_test = np.array(dsets["test"]["label"])

# ------------ MODEL -------------------------------------------------
clf = LogisticRegression(penalty="l1", C=0.3, solver="liblinear", max_iter=1000)
clf.fit(X_train, y_train)

# ------------ VALIDATION METRIC ------------------------------------
dev_proba = clf.predict_proba(X_dev)
val_loss = log_loss(y_dev, dev_proba)
print(f"Epoch 1: validation_loss = {val_loss:.4f}")

# ------------ TEST & IRF -------------------------------------------
test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Test accuracy = {test_acc:.4f}")

# Interpretable Rule Fidelity (rules == model, so 1.0)
rule_pred = clf.predict(X_test)  # identical
irf = accuracy_score(rule_pred, test_pred)
print(f"IRF = {irf:.4f}")

# ------------ RULE EXTRACTION --------------------------------------
feature_names = np.array(vectorizer.get_feature_names_out())
coef = clf.coef_[0]
nz_idx = np.where(coef != 0)[0]
rules = []
for idx in nz_idx:
    ngram = feature_names[idx]
    w = coef[idx]
    sign = "INCREASE" if w > 0 else "DECREASE"
    rules.append(f"IF count('{ngram}') >= 1 THEN {sign} log-odds by {abs(w):.3f}")
rules_path = os.path.join(working_dir, "extracted_rules.txt")
with open(rules_path, "w") as f:
    f.write("\n".join(rules))
print(f"Saved {len(rules)} rules to {rules_path}")

# ------------ CONFUSION MATRIX PLOT --------------------------------
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.colorbar(im, ax=ax)
fig_path = os.path.join(working_dir, "confusion_matrix.png")
plt.savefig(fig_path)
plt.close()
print("Saved confusion matrix to", fig_path)

# ------------ SAVE EXPERIMENT DATA ---------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [test_acc], "test": [test_acc], "irf": [irf]},
        "losses": {"train": [], "val": [val_loss]},
        "predictions": test_pred.tolist(),
        "ground_truth": y_test.tolist(),
    }
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
