import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset

# --------------------------------------------------------------------
# GPU / CPU handling (mandatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------------------------------------------------------


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
        print("Loaded SPR_BENCH from", possible_path)
        return spr
    except Exception as e:
        print("Falling back to synthetic toy dataset:", e)
        rng = np.random.default_rng(0)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                length = rng.integers(4, 8)
                seq = "".join(rng.choice(vocab, size=length))
                labels.append(int(seq.count("A") % 2 == 0))
                seqs.append(seq)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(1000), dev=gen(300), test=gen(300))


dsets = get_dataset()

# ---------------------- PREPROCESSING -------------------------------
chars = sorted({ch for split in dsets for s in dsets[split]["sequence"] for ch in s})
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)
print("Character vocab:", chars)


def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    for ch in seq:
        v[char2idx[ch]] += 1.0
    return v


def vectorise_split(name):
    X = np.stack([seq_to_vec(s) for s in dsets[name]["sequence"]])
    y = np.array(dsets[name]["label"], dtype=np.float32)
    return X, y


X_train, y_train = vectorise_split("train")
X_dev, y_dev = vectorise_split("dev")
X_test, y_test = vectorise_split("test")

# ---------------------- DATALOADERS ---------------------------------
batch_size = 64
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=batch_size,
    shuffle=True,
)
dev_loader = DataLoader(
    TensorDataset(torch.tensor(X_dev), torch.tensor(y_dev)), batch_size=len(y_dev)
)


# ---------------------- MODEL ---------------------------------------
class MLP(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = MLP(V).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------- TRAINING LOOP -------------------------------
epochs = 20
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "IRF": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
    }
}


def evaluate(loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    loss = log_loss(labels.numpy(), probs, labels=[0, 1])
    acc = accuracy_score(labels.numpy(), preds)
    return loss, acc, probs, preds


for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * yb.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == yb.long()).sum().item()
        total += yb.size(0)
    train_acc = correct / total
    train_loss = epoch_loss / total
    val_loss, val_acc, _, _ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )

# ---------------------- TEST EVALUATION -----------------------------
model.eval()
with torch.no_grad():
    logits_test = model(torch.tensor(X_test).to(device)).cpu()
probs_test = torch.sigmoid(logits_test).numpy()
mlp_pred = (probs_test >= 0.5).astype(int)
test_acc = accuracy_score(y_test, mlp_pred)
print(f"\nNeural model test accuracy: {test_acc:.4f}")

# ---------------------- SURROGATE RULE TREE -------------------------
tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(
    X_train, (model(torch.tensor(X_train).to(device)).sigmoid().cpu() >= 0.5).int()
)
tree_pred = tree.predict(X_test)
IRF = accuracy_score(mlp_pred, tree_pred)
experiment_data["SPR_BENCH"]["metrics"]["IRF"] = [IRF]
print(f"Interpretable Rule Fidelity (IRF) on test: {IRF:.4f}")


# Rule extraction
def extract_rules(tree_clf, feature_names):
    tree_ = tree_clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, cur):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            thresh = tree_.threshold[node]
            recurse(tree_.children_left[node], cur + [f"{name} <= {thresh:.1f}"])
            recurse(tree_.children_right[node], cur + [f"{name} > {thresh:.1f}"])
        else:
            pred = tree_.value[node][0]
            label = np.argmax(pred)
            rule = " AND ".join(cur) if cur else "TRUE"
            rules.append(f"IF {rule} THEN model_label={label}")

    recurse(0, [])
    return rules


rules = extract_rules(tree, chars)
with open(os.path.join(working_dir, "extracted_rules.txt"), "w") as f:
    f.write("\n".join(rules))
print(f"Saved {len(rules)} rules to extracted_rules.txt")

# ---------------------- CONFUSION MATRIX ----------------------------
cm = confusion_matrix(y_test, mlp_pred)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Neural Confusion")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.colorbar(im, ax=ax)
fig_path = os.path.join(working_dir, "confusion_matrix.png")
plt.savefig(fig_path)
plt.close()
print("Saved confusion matrix to", fig_path)

# ---------------------- SAVE EXPERIMENT DATA ------------------------
experiment_data["SPR_BENCH"]["predictions"] = mlp_pred.tolist()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
