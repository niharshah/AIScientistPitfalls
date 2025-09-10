import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, numpy as np, torch, torch.nn as nn, torch.utils.data as td
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, log_loss
from datasets import load_dataset, Dataset, DatasetDict

# -------------------- device handling ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- load (or create) dataset -------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):  # treat csv as single split
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


def get_dataset() -> DatasetDict:
    real_path = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        ds = load_spr_bench(real_path)
        print("Loaded SPR_BENCH from", real_path)
        return ds
    except Exception as e:
        print("Real dataset not found – generating synthetic parity data.")
        rng = np.random.default_rng(0)
        vocab = list("ABC")

        def make(n):
            seq, lab = [], []
            for i in range(n):
                L = int(rng.integers(4, 9))
                s = "".join(rng.choice(vocab, size=L))
                seq.append(s)
                lab.append(int(s.count("A") % 2 == 0))
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seq, "label": lab}
            )

        return DatasetDict(train=make(1200), dev=make(400), test=make(400))


dsets = get_dataset()

# -------------------- feature extraction -------------------------------
# build unigram and bigram vocab
unichars = set()
bigrams = set()
for split in dsets:
    for seq in dsets[split]["sequence"]:
        unichars.update(seq)
        bigrams.update([seq[i : i + 2] for i in range(len(seq) - 1)])
unichars = sorted(list(unichars))
bigrams = sorted(list(bigrams))
u2i = {c: i for i, c in enumerate(unichars)}
b2i = {b: i for i, b in enumerate(bigrams)}
F = len(unichars) + len(bigrams)
print(f"Feature size: {F}  (unigrams {len(unichars)}, bigrams {len(bigrams)})")


def seq_to_vec(seq: str) -> np.ndarray:
    v = np.zeros(F, dtype=np.float32)
    for ch in seq:
        v[u2i[ch]] += 1.0
    for i in range(len(seq) - 1):
        bg = seq[i : i + 2]
        if bg in b2i:
            v[len(unichars) + b2i[bg]] += 1.0
    v /= len(seq)  # normalise by length
    return v


def vectorise_split(name):
    X = np.stack([seq_to_vec(s) for s in dsets[name]["sequence"]])
    y = np.array(dsets[name]["label"], dtype=np.int64)
    return X, y


X_train, y_train = vectorise_split("train")
X_dev, y_dev = vectorise_split("dev")
X_test, y_test = vectorise_split("test")


# -------------------- torch dataset -----------------------------------
class NpDataset(td.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


batch_size = 128
train_loader = td.DataLoader(
    NpDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = td.DataLoader(NpDataset(X_dev, y_dev), batch_size=512)


# -------------------- MLP model ---------------------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(F).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

# -------------------- training loop -----------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "IRF": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
    }
}


def evaluate(loader):
    model.eval()
    ys, preds, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            losses.append(loss.item() * len(batch["y"]))
            probs = torch.softmax(logits, 1)
            pred = probs.argmax(1).cpu().numpy()
            ys.append(batch["y"].cpu().numpy())
            preds.append(pred)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    return np.sum(losses) / len(ys), accuracy_score(ys, preds)


epochs = 15
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch["y"])
    train_loss = epoch_loss / len(X_train)
    val_loss, val_acc = evaluate(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        1 - train_loss
    )  # placeholder metric
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)

# -------------------- test accuracy -----------------------------------
_, test_acc = evaluate(td.DataLoader(NpDataset(X_test, y_test), batch_size=512))
print(f"Neural model test accuracy: {test_acc:.4f}")

# -------------------- decision tree surrogate & IRF -------------------
surrogate = DecisionTreeClassifier(max_depth=6, random_state=0)
surrogate.fit(X_train, model(torch.tensor(X_train).to(device)).argmax(1).cpu().numpy())
tree_pred = surrogate.predict(X_test)
nn_pred = model(torch.tensor(X_test).to(device)).argmax(1).cpu().numpy()
irf = accuracy_score(nn_pred, tree_pred)
print(f"Interpretable Rule Fidelity (IRF): {irf:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["IRF"].append(irf)
experiment_data["SPR_BENCH"]["predictions"] = nn_pred.tolist()


# -------------------- rule extraction ---------------------------------
def tree_to_rules(tree, feat_names):
    t = tree.tree_
    fn = [
        feat_names[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in t.feature
    ]
    rules = []

    def recurse(node, conds):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            thr = t.threshold[node]
            for dir, child in [
                ("<= ", t.children_left[node]),
                ("> ", t.children_right[node]),
            ]:
                recurse(child, conds + [f"{fn[node]} {dir}{thr:.1f}"])
        else:
            pred = np.argmax(t.value[node][0])
            rule = " and ".join(conds) if conds else "TRUE"
            rules.append(f"IF {rule} THEN label={pred}")

    recurse(0, [])
    return rules


feature_names = unichars + bigrams
rules = tree_to_rules(surrogate, feature_names)
with open(os.path.join(working_dir, "rules.txt"), "w") as f:
    f.write("\n".join(rules))
print(f"Saved {len(rules)} rules to working/rules.txt")

# -------------------- save experiment data ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
