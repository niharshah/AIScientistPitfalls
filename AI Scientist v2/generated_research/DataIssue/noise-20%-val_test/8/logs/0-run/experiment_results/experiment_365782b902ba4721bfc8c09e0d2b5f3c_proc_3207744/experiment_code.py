import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, json, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, confusion_matrix
from datasets import load_dataset, Dataset, DatasetDict

# ---------------- device handling -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- data loading --------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def get_dataset() -> DatasetDict:
    real_path = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        d = load_spr_bench(real_path)
        print("Loaded SPR_BENCH from", real_path)
        return d
    except Exception as e:
        print("SPR_BENCH not found, generating synthetic dataset.")
        rng = np.random.default_rng(0)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                L = rng.integers(4, 9)
                s = "".join(rng.choice(vocab, size=L))
                lab = int(s.count("A") % 2 == 0)
                seqs.append(s)
                labels.append(lab)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(6000), dev=gen(2000), test=gen(2000))


dsets = get_dataset()

# --------------- vectorisation --------------------------------------
chars = sorted({c for split in dsets for s in dsets[split]["sequence"] for c in s})
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)


def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    for ch in seq:
        if ch in char2idx:
            v[char2idx[ch]] += 1.0
    return v / len(seq)  # normalised bag-of-chars


def vectorise(split):
    X = np.stack([seq_to_vec(s) for s in dsets[split]["sequence"]])
    y = np.array(dsets[split]["label"])
    return X, y


X_train, y_train = vectorise("train")
X_dev, y_dev = vectorise("dev")
X_test, y_test = vectorise("test")


# torch datasets
def loader(X, y, batch=128, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


train_loader = loader(X_train, y_train, shuffle=True)
dev_loader = loader(X_dev, y_dev)


# --------------- model ----------------------------------------------
class MLP(nn.Module):
    def __init__(self, inp, hidden=64, out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(V).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# --------------- helper: rule extraction ----------------------------
def extract_tree_rules(tree, feature_names):
    tree_ = tree.tree_
    fn = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]
    rules = []

    def rec(node, cond):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = fn[node]
            thr = tree_.threshold[node]
            rec(tree_.children_left[node], cond + [f"{name}<={thr:.2f}"])
            rec(tree_.children_right[node], cond + [f"{name}>{thr:.2f}"])
        else:
            pred = np.argmax(tree_.value[node][0])
            rule = " AND ".join(cond) if cond else "TRUE"
            rules.append(f"IF {rule} THEN label={pred}")

    rec(0, [])
    return rules


# --------------- training loop --------------------------------------
EPOCHS = 10
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "IRF": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
    }
}

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * yb.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total
    train_loss = epoch_loss / total

    # validation
    model.eval()
    with torch.no_grad():
        v_loss, v_correct, v_total = 0.0, 0, 0
        dev_probs = []
        for xb, yb in dev_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            v_loss += loss.item() * yb.size(0)
            v_correct += (out.argmax(1) == yb).sum().item()
            v_total += yb.size(0)
            dev_probs.append(out.softmax(1).cpu().numpy())
        val_acc = v_correct / v_total
        val_loss = v_loss / v_total

    # surrogate tree & IRF
    with torch.no_grad():
        train_logits = model(torch.tensor(X_train, dtype=torch.float32).to(device))
        pseudo = train_logits.argmax(1).cpu().numpy()
    tree = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, pseudo)
    dev_tree_pred = tree.predict(X_dev)
    with torch.no_grad():
        dev_nn_pred = (
            model(torch.tensor(X_dev, dtype=torch.float32).to(device))
            .argmax(1)
            .cpu()
            .numpy()
        )
    irf = (dev_tree_pred == dev_nn_pred).mean()

    # log
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.3f} | IRF={irf:.3f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["IRF"].append(irf)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

# --------------- final evaluation -----------------------------------
model.eval()
with torch.no_grad():
    test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    test_pred = test_logits.argmax(1).cpu().numpy()
test_acc = accuracy_score(y_test, test_pred)
print("Neural model test accuracy:", test_acc)

# final surrogate for rules and IRF on test
tree_final = DecisionTreeClassifier(max_depth=5, random_state=0).fit(
    X_train,
    model(torch.tensor(X_train, dtype=torch.float32).to(device))
    .argmax(1)
    .cpu()
    .numpy(),
)
test_tree_pred = tree_final.predict(X_test)
IRF_test = (test_tree_pred == test_pred).mean()
print("Test IRF:", IRF_test)

# save rules
rules = extract_tree_rules(tree_final, chars)
rules_path = os.path.join(working_dir, "extracted_rules.txt")
with open(rules_path, "w") as f:
    f.write("\n".join(rules))
print(f"Saved {len(rules)} rules to", rules_path)

# confusion matrix
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Pred")
ax.set_ylabel("True")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.colorbar(im, ax=ax)
cm_path = os.path.join(working_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print("Saved confusion matrix to", cm_path)

# save experiment data
experiment_data["SPR_BENCH"]["predictions"] = test_pred.tolist()
experiment_data["SPR_BENCH"]["metrics"]["test_acc"] = test_acc
experiment_data["SPR_BENCH"]["metrics"]["IRF_test"] = IRF_test
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
