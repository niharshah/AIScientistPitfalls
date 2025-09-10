import os, pathlib, time, copy, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from typing import Dict
from datasets import load_dataset, DatasetDict

# ---------- HOUSEKEEPING ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- EXPERIMENT DATA STORE ----------
experiment_data: Dict = {
    "EPOCHS": {
        "SPR_BENCH": {
            "epochs_list": [],
            "metrics": {"train_acc": [], "val_acc": [], "val_loss": []},
            "losses": {"train": [], "val": []},
            "test_acc": [],
            "fidelity": [],
            "predictions": [],
            "ground_truth": [],
            "rule_preds": [],
        }
    }
}


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
    print("SPR_BENCH not found, creating synthetic toy data.")
    seqs = ["ABAB", "BABA", "AAAA", "BBBB"]
    labels = [0, 0, 1, 1]
    from datasets import Dataset

    d = Dataset.from_dict({"id": list(range(4)), "sequence": seqs, "label": labels})
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
input_dim, num_classes = X_train.shape[1], len(
    set(y_train.tolist() + y_val.tolist() + y_test.tolist())
)
print(f"Input dim {input_dim}, #classes {num_classes}")


# ---------- DATASET ----------
class SparseNPDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.X[idx].toarray()).squeeze(0),
            "y": torch.tensor(self.y[idx]),
        }


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


train_ds, val_ds, test_ds = (
    SparseNPDataset(X_train, y_train),
    SparseNPDataset(X_val, y_val),
    SparseNPDataset(X_test, y_test),
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

# ---------- HYPER-PARAMETER GRID ----------
EPOCH_OPTIONS = [5, 20, 50]  # simple grid for epoch count
best_val_acc, best_state = -1.0, None


# ---------- TRAINING FUNCTION ----------
def train_for_epochs(num_epochs):
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_acc_hist, val_acc_hist, val_loss_hist, train_loss_hist = [], [], [], []
    for epoch in range(1, num_epochs + 1):
        # train
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
            preds = out.argmax(1)
            correct += (preds == batch["y"]).sum().item()
            total += batch["y"].size(0)
        train_acc_hist.append(correct / total)
        train_loss_hist.append(running_loss / total)
        # validation
        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                v_loss += loss.item() * batch["y"].size(0)
                preds = out.argmax(1)
                v_correct += (preds == batch["y"]).sum().item()
                v_total += batch["y"].size(0)
        val_acc_hist.append(v_correct / v_total)
        val_loss_hist.append(v_loss / v_total)
    return model, train_acc_hist, val_acc_hist, val_loss_hist, train_loss_hist


# ---------- GRID SEARCH ----------
for EPOCHS in EPOCH_OPTIONS:
    print(f"\n=== Training for {EPOCHS} epochs ===")
    model, tr_acc, vl_acc, vl_loss, tr_loss = train_for_epochs(EPOCHS)
    # store
    store = experiment_data["EPOCHS"]["SPR_BENCH"]
    store["epochs_list"].append(EPOCHS)
    store["metrics"]["train_acc"].append(tr_acc)
    store["metrics"]["val_acc"].append(vl_acc)
    store["metrics"]["val_loss"].append(vl_loss)
    store["losses"]["train"].append(tr_loss)
    # keep best
    if vl_acc[-1] > best_val_acc:
        best_val_acc = vl_acc[-1]
        best_state = copy.deepcopy(model.state_dict())

# ---------- EVALUATE BEST MODEL ----------
best_model = nn.Sequential(
    nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
).to(device)
best_model.load_state_dict(best_state)
best_model.eval()


def predict(loader):
    preds_all, ys_all = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = best_model(x)
            preds_all.append(out.argmax(1).cpu().numpy())
            ys_all.append(batch["y"].numpy())
    return np.concatenate(preds_all), np.concatenate(ys_all)


test_preds, test_gt = predict(test_loader)
test_acc = (test_preds == test_gt).mean()
print(
    f"\nBest validation accuracy {best_val_acc:.4f}  --> test accuracy {test_acc:.4f}"
)

# ---------- RULE EXTRACTION ----------
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(
    X_train,
    best_model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy(),
)
rule_preds = tree.predict(X_test)
fidelity = (rule_preds == test_preds).mean()
print(f"Rule fidelity: {fidelity:.4f}")
fagm = np.sqrt(test_acc * fidelity)
print(f"FAGM: {fagm:.4f}")

# ---------- SAVE RESULTS ----------
store = experiment_data["EPOCHS"]["SPR_BENCH"]
store["test_acc"].append(test_acc)
store["fidelity"].append(fidelity)
store["predictions"] = test_preds
store["ground_truth"] = test_gt
store["rule_preds"] = rule_preds
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
