import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from typing import Dict
from datasets import load_dataset, DatasetDict

# ---------- EXPERIMENT DATA STORE ----------
experiment_data: Dict = {
    "learning_rate": {"SPR_BENCH": {}}  # one sub-dict per tested lr will be inserted
}

# ---------- HOUSEKEEPING ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- DATA LOADING ----------
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


DATA_PATH = pathlib.Path("./SPR_BENCH")
try:
    spr = load_spr_bench(DATA_PATH)
except FileNotFoundError:
    # fallback tiny synthetic dataset
    print("SPR_BENCH not found, creating synthetic toy data.")
    seqs = ["ABAB", "BABA", "AAAA", "BBBB"]
    labels = [0, 0, 1, 1]
    tiny = {"id": list(range(4)), "sequence": seqs, "label": labels}
    from datasets import Dataset

    d = Dataset.from_dict(tiny)
    spr = DatasetDict(train=d, dev=d, test=d)

# ---------- VECTORISATION ----------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
vectorizer.fit(spr["train"]["sequence"])


def vectorise(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32)
    y = np.array(split["label"], dtype=np.int64)
    return X, y


X_train, y_train = vectorise(spr["train"])
X_val, y_val = vectorise(spr["dev"])
X_test, y_test = vectorise(spr["test"])
input_dim = X_train.shape[1]
num_classes = len(set(y_train.tolist() + y_val.tolist() + y_test.tolist()))
print(f"Input dim {input_dim}, #classes {num_classes}")


# ---------- DATASET WRAPPER ----------
class SparseNPDataset(Dataset):
    def __init__(self, X_csr, y):
        self.X = X_csr
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].toarray()).squeeze(0)
        return {"x": x, "y": torch.tensor(self.y[idx])}


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


train_ds = SparseNPDataset(X_train, y_train)
val_ds = SparseNPDataset(X_val, y_val)
test_ds = SparseNPDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- TRAINING UTILITIES ----------
def make_model():
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    ).to(device)


criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optim):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optim.step()
        running_loss += loss.item() * batch["y"].size(0)
        preds = outputs.argmax(1)
        correct += (preds == batch["y"]).sum().item()
        total += batch["y"].size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        running_loss += loss.item() * batch["y"].size(0)
        preds = outputs.argmax(1)
        correct += (preds == batch["y"]).sum().item()
        total += batch["y"].size(0)
    return running_loss / total, correct / total


# ---------- LEARNING RATE SWEEP ----------
lrs = [1e-3, 5e-4, 1e-4, 5e-5]
EPOCHS = 5
best_state, best_val_acc, best_val_loss, best_lr = None, -1.0, float("inf"), None

for lr in lrs:
    tag = str(lr)
    print(f"\n=== Training with lr={lr} ===")
    experiment_data["learning_rate"]["SPR_BENCH"][tag] = {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": []},
        "losses": {"train": []},
    }

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer)
        v_loss, v_acc = eval_epoch(model, val_loader)

        ed = experiment_data["learning_rate"]["SPR_BENCH"][tag]
        ed["losses"]["train"].append(t_loss)
        ed["metrics"]["train_acc"].append(t_acc)
        ed["metrics"]["val_loss"].append(v_loss)
        ed["metrics"]["val_acc"].append(v_acc)

        print(f"Epoch {epoch}: val_loss={v_loss:.4f}, val_acc={v_acc:.4f}")

    # keep best weights
    if (v_acc > best_val_acc) or (v_acc == best_val_acc and v_loss < best_val_loss):
        best_val_acc, best_val_loss, best_lr = v_acc, v_loss, lr
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

print(f"\nBest lr: {best_lr} with val_acc={best_val_acc:.4f}")

# ---------- TEST EVALUATION WITH BEST MODEL ----------
best_model = make_model()
best_model.load_state_dict(best_state)
test_loss, test_acc = eval_epoch(best_model, test_loader)
print(f"Test accuracy with best lr: {test_acc:.4f}")


def predict_loader(loader, mdl):
    preds_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            outputs = mdl(x)
            preds_all.append(outputs.argmax(1).cpu().numpy())
            y_all.append(batch["y"].numpy())
    return np.concatenate(preds_all), np.concatenate(y_all)


test_preds, test_gt = predict_loader(test_loader, best_model)

# store final results
experiment_data["learning_rate"]["SPR_BENCH"]["best_lr"] = best_lr
experiment_data["learning_rate"]["SPR_BENCH"]["test_acc"] = test_acc
experiment_data["learning_rate"]["SPR_BENCH"]["predictions"] = test_preds
experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"] = test_gt

# ---------- RULE EXTRACTION (Decision Tree Distillation) ----------
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(
    X_train,
    best_model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy(),
)
rule_preds = tree.predict(X_test)
fidelity = (rule_preds == test_preds).mean()
experiment_data["learning_rate"]["SPR_BENCH"]["rule_preds"] = rule_preds
experiment_data["learning_rate"]["SPR_BENCH"]["rule_fidelity"] = fidelity
fagm = np.sqrt(test_acc * fidelity)
experiment_data["learning_rate"]["SPR_BENCH"]["fagm"] = fagm
print(f"Rule fidelity: {fidelity:.4f}  |  FAGM: {fagm:.4f}")

# ---------- SAVE METRICS ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
