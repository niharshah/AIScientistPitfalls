import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List
from datasets import load_dataset, DatasetDict

# ---------- EXPERIMENT DATA STORE ----------
experiment_data: Dict = {
    "num_hidden_layers": {
        "SPR_BENCH": {
            "configs": [],  # e.g. ["layers_1", "layers_2", ...]
            "metrics": {"train_acc": [], "val_acc": [], "val_loss": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
            "rule_preds": [],
            "fagm": [],
        }
    }
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
except (FileNotFoundError, Exception):
    print("SPR_BENCH not found, creating synthetic toy data.")
    seqs = ["ABAB", "BABA", "AAAA", "BBBB"]
    labels = [0, 0, 1, 1]
    from datasets import Dataset

    toy = Dataset.from_dict({"id": list(range(4)), "sequence": seqs, "label": labels})
    spr = DatasetDict(train=toy, dev=toy, test=toy)

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
        self.X, self.y = X_csr, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].toarray()).squeeze(0)
        return {"x": x, "y": torch.tensor(self.y[idx])}


train_ds, val_ds, test_ds = (
    SparseNPDataset(X_train, y_train),
    SparseNPDataset(X_val, y_val),
    SparseNPDataset(X_test, y_test),
)


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- TRAIN / EVAL HELPERS ----------
def build_mlp(depth: int, in_dim: int, hid: int, n_classes: int) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(in_dim, hid), nn.ReLU()]
    for _ in range(depth - 1):  # already had first hidden layer
        layers += [nn.Linear(hid, hid), nn.ReLU()]
    layers += [nn.Linear(hid, n_classes)]
    return nn.Sequential(*layers).to(device)


def eval_loader(model, loader, criterion):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            preds = out.argmax(1)
            loss_sum += loss.item() * batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            total += batch["y"].size(0)
    return loss_sum / total, correct / total


def predict_loader(model, loader):
    preds_all, y_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)
            preds_all.append(out.argmax(1).cpu().numpy())
            y_all.append(batch["y"].numpy())
    return np.concatenate(preds_all), np.concatenate(y_all)


# ---------- HYPERPARAMETER SWEEP ----------
EPOCHS = 5
hidden_dim = 128
depth_options = [1, 2, 3]

store = experiment_data["num_hidden_layers"]["SPR_BENCH"]
store["ground_truth"] = y_test  # same for every config

for depth in depth_options:
    config_name = f"layers_{depth}"
    print(f"\n=== Training model with {depth} hidden layer(s) ===")
    store["configs"].append(config_name)

    model = build_mlp(depth, input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tr_acc_hist, val_acc_hist, val_loss_hist, tr_loss_hist = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = correct = total = 0
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
        tr_loss = running_loss / total
        tr_acc = correct / total
        val_loss, val_acc = eval_loader(model, val_loader, criterion)

        tr_loss_hist.append(tr_loss)
        tr_acc_hist.append(tr_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        print(f"Epoch {epoch}: train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

    # store per-config history
    store["losses"]["train"].append(tr_loss_hist)
    store["metrics"]["train_acc"].append(tr_acc_hist)
    store["metrics"]["val_acc"].append(val_acc_hist)
    store["metrics"]["val_loss"].append(val_loss_hist)

    # --- Test evaluation ---
    test_preds, test_gt = predict_loader(model, test_loader)
    test_acc = (test_preds == test_gt).mean()
    print(f"Test accuracy ({config_name}): {test_acc:.4f}")
    store["predictions"].append(test_preds)

    # --- Rule extraction / fidelity ---
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(
        X_train,
        model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy(),
    )
    rule_preds = tree.predict(X_test)
    fidelity = (rule_preds == test_preds).mean()
    fagm = np.sqrt(test_acc * fidelity)
    print(f"Fidelity: {fidelity:.4f} | FAGM: {fagm:.4f}")
    store["rule_preds"].append(rule_preds)
    store["fagm"].append(fagm)

# ---------- SAVE METRICS ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to 'working/experiment_data.npy'")
