import os, pathlib, numpy as np, torch, torch.nn as nn, random, time
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from typing import Dict
from datasets import load_dataset, DatasetDict

# ---------- HOUSEKEEPING ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data: Dict = {
    "weight_decay": {
        "SPR_BENCH": {
            "wds": [],  # weight-decay values tried
            "metrics": {"train_acc": [], "val_acc": [], "val_loss": []},
            "losses": {"train": []},
            "test_acc": [],
            "fidelity": [],
            "fagm": [],
            "predictions": [],
            "ground_truth": [],
            "rule_preds": [],
        }
    }
}


# ---------- DATA LOADING ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
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

    tiny = {"id": list(range(4)), "sequence": seqs, "label": labels}
    d = Dataset.from_dict(tiny)
    spr = DatasetDict(train=d, dev=d, test=d)

# ---------- VECTORIZATION ----------
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
        return {
            "x": torch.from_numpy(self.X[idx].toarray()).squeeze(0),
            "y": torch.tensor(self.y[idx]),
        }


train_ds = SparseNPDataset(X_train, y_train)
val_ds = SparseNPDataset(X_val, y_val)
test_ds = SparseNPDataset(X_test, y_test)


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- TRAIN / EVAL FUNCTIONS ----------
def build_model():
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
    )
    return model.to(device)


def eval_loader(model, loader, criterion):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"])
            l = criterion(out, batch["y"])
            loss += l.item() * batch["y"].size(0)
            pred = out.argmax(1)
            correct += (pred == batch["y"]).sum().item()
            total += batch["y"].size(0)
    return loss / total, correct / total


def predict_loader(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x).argmax(1).cpu().numpy()
            preds.append(out)
            gts.append(batch["y"].numpy())
    return np.concatenate(preds), np.concatenate(gts)


# ---------- HYPERPARAMETER TUNING ----------
EPOCHS = 5
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
            pred = out.argmax(1)
            correct += (pred == batch["y"]).sum().item()
            total += batch["y"].size(0)
        train_acc = correct / total
        val_loss, val_acc = eval_loader(model, val_loader, criterion)

        # store epoch metrics
        experiment_data["weight_decay"]["SPR_BENCH"]["losses"]["train"].append(
            running_loss / total
        )
        experiment_data["weight_decay"]["SPR_BENCH"]["metrics"]["train_acc"].append(
            train_acc
        )
        experiment_data["weight_decay"]["SPR_BENCH"]["metrics"]["val_loss"].append(
            val_loss
        )
        experiment_data["weight_decay"]["SPR_BENCH"]["metrics"]["val_acc"].append(
            val_acc
        )
        print(
            f"Epoch {epoch}: train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    # --- TEST + DISTILLATION ---
    test_preds, test_gt = predict_loader(model, test_loader)
    test_acc = (test_preds == test_gt).mean()
    tree = DecisionTreeClassifier(max_depth=5, random_state=seed)
    tree.fit(
        X_train,
        model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy(),
    )
    rule_preds = tree.predict(X_test)
    fidelity = (rule_preds == test_preds).mean()
    fagm = np.sqrt(test_acc * fidelity)

    # --- SAVE RESULTS ---
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
    ed["wds"].append(wd)
    ed["test_acc"].append(test_acc)
    ed["fidelity"].append(fidelity)
    ed["fagm"].append(fagm)
    ed["predictions"].append(test_preds)
    ed["ground_truth"].append(test_gt)
    ed["rule_preds"].append(rule_preds)

    print(
        f"weight_decay={wd}: test_acc={test_acc:.4f}, fidelity={fidelity:.4f}, FAGM={fagm:.4f}"
    )

# ---------- PERSIST ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
