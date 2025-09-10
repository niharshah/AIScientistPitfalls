import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Device handling (mandatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------------------------------------------------------


########################  DATA  ######################################
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def get_dataset() -> DatasetDict:
    root = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        d = load_spr_bench(root)
        print("Loaded SPR_BENCH from", root)
        return d
    except Exception:
        print("SPR_BENCH not found – generating small synthetic parity dataset.")
        rng = np.random.default_rng(0)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                length = rng.integers(4, 9)
                s = "".join(rng.choice(vocab, size=length))
                lbl = int(s.count("A") % 2 == 0)
                seqs.append(s)
                labels.append(lbl)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(1000), dev=gen(300), test=gen(300))


dsets = get_dataset()

#####################  FEATURE ENGINEERING  ##########################
# Collect unigrams & bigrams from train split
unigrams = set()
bigrams = set()
for s in dsets["train"]["sequence"]:
    unigrams.update(list(s))
    bigrams.update([s[i] + s[i + 1] for i in range(len(s) - 1)])
unigrams = sorted(list(unigrams))
bigrams = sorted(list(bigrams))
feat2idx = {tok: i for i, tok in enumerate(unigrams + bigrams)}
V = len(feat2idx)
print(f"Feature dim = {V} ({len(unigrams)} unigrams + {len(bigrams)} bigrams)")


def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    for ch in seq:
        v[feat2idx[ch]] += 1.0
    for i in range(len(seq) - 1):
        bg = seq[i] + seq[i + 1]
        if bg in feat2idx:
            v[feat2idx[bg]] += 1.0
    return v


def vectorise_split(split):
    X = np.stack([seq_to_vec(s) for s in dsets[split]["sequence"]])
    y = np.array(dsets[split]["label"], dtype=np.int64)
    return X, y


X_train, y_train = vectorise_split("train")
X_dev, y_dev = vectorise_split("dev")
X_test, y_test = vectorise_split("test")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=256,
    shuffle=True,
)


###########################  MODEL  ##################################
class MLP(nn.Module):
    def __init__(self, dim_in, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(V).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#####################  TRACKING STRUCTURE  ###########################
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "test_acc": None, "IRF": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
    }
}

############################ TRAIN ###################################
EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_y.size(0)
        preds = out.argmax(-1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    with torch.no_grad():
        dev_logits = model(torch.tensor(X_dev).to(device))
        dev_loss = criterion(dev_logits, torch.tensor(y_dev).to(device)).item()
        dev_pred = dev_logits.argmax(-1).cpu().numpy()
        val_acc = (dev_pred == y_dev).mean()

    print(f"Epoch {epoch}: validation_loss = {dev_loss:.4f}")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(dev_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)

############################ TEST ####################################
model.eval()
with torch.no_grad():
    test_logits = model(torch.tensor(X_test).to(device))
    test_pred = test_logits.argmax(-1).cpu().numpy()
test_acc = (test_pred == y_test).mean()
experiment_data["SPR_BENCH"]["test_predictions"] = test_pred.tolist()
experiment_data["SPR_BENCH"]["metrics"]["test_acc"] = test_acc
print(f"Test accuracy = {test_acc:.4f}")

#####################  SURROGATE TREE & IRF  #########################
clf_surrogate = DecisionTreeClassifier(max_depth=5, random_state=0)
model_pred_train = model(torch.tensor(X_train).to(device)).argmax(-1).cpu().numpy()
clf_surrogate.fit(X_train, model_pred_train)

tree_pred_test = clf_surrogate.predict(X_test)
irf = (tree_pred_test == test_pred).mean()
experiment_data["SPR_BENCH"]["metrics"]["IRF"] = irf
print(f"Interpretable Rule Fidelity (IRF) = {irf:.4f}")


# --------- save human-readable rules ----------
def tree_to_rules(tree, feat_names):
    tree_ = tree.tree_
    feature_name = [
        feat_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    def recurse(node, conds):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            thr = tree_.threshold[node]
            recurse(tree_.children_left[node], conds + [f"{name} <= {thr:.1f}"])
            recurse(tree_.children_right[node], conds + [f"{name} > {thr:.1f}"])
        else:
            proba = tree_.value[node][0]
            pred = np.argmax(proba)
            paths.append(
                "IF "
                + (" AND ".join(conds) if conds else "TRUE")
                + f" THEN label={pred}"
            )

    recurse(0, [])
    return paths


rules = tree_to_rules(clf_surrogate, unigrams + bigrams)
with open(os.path.join(working_dir, "surrogate_rules.txt"), "w") as f:
    f.write("\n".join(rules))
print(f"Saved {len(rules)} rules to surrogate_rules.txt")

####################  CONFUSION MATRIX PLOT  #########################
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.colorbar(im, ax=ax)
plt.savefig(os.path.join(working_dir, "confusion_matrix.png"))
plt.close()

######################## SAVE EXPERIMENT DATA ########################
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiment data saved to experiment_data.npy")
