import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from typing import Dict
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ---------- HOUSEKEEPING ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- EXPERIMENT DATA STORE ----------
experiment_data: Dict = {"batch_size": {"SPR_BENCH": {}}}


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


DATA_PATH = pathlib.Path("./SPR_BENCH")  # adjust if needed
try:
    spr = load_spr_bench(DATA_PATH)
except FileNotFoundError:
    # fallback tiny synthetic dataset
    print("SPR_BENCH not found, creating synthetic toy data.")
    seqs = ["ABAB", "BABA", "AAAA", "BBBB"]
    labels = [0, 0, 1, 1]
    tiny = {"id": list(range(4)), "sequence": seqs, "label": labels}
    d = HFDataset.from_dict(tiny)
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


train_ds = SparseNPDataset(X_train, y_train)
val_ds = SparseNPDataset(X_val, y_val)
test_ds = SparseNPDataset(X_test, y_test)


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


# ---------- HYPERPARAMETER SWEEP ----------
BATCH_SIZES = [32, 64, 128, 256, 512]
EPOCHS = 5

for bs in BATCH_SIZES:
    print(f"\n========== Training with batch_size={bs} ==========")
    # data loaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate)

    # model, loss, opt
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage dict for this bs
    exp_bs = {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
        "rule_preds": [],
        "test_acc": None,
        "fidelity": None,
        "fagm": None,
    }

    # ---------- TRAIN LOOP ----------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch["x"])
            loss = criterion(outputs, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["y"].size(0)
            preds = outputs.argmax(1)
            correct += (preds == batch["y"]).sum().item()
            total += batch["y"].size(0)
        train_acc = correct / total
        exp_bs["losses"]["train"].append(running_loss / total)
        exp_bs["metrics"]["train_acc"].append(train_acc)

        # validation
        model.eval()
        val_loss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["x"])
                loss = criterion(outputs, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                preds = outputs.argmax(1)
                vcorrect += (preds == batch["y"]).sum().item()
                vtotal += batch["y"].size(0)
        val_loss /= vtotal
        val_acc = vcorrect / vtotal
        exp_bs["metrics"]["val_loss"].append(val_loss)
        exp_bs["metrics"]["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch}: train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

    # ---------- TEST EVALUATION ----------
    def predict_loader(loader):
        preds_all, y_all = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                outputs = model(x)
                preds = outputs.argmax(1).cpu().numpy()
                preds_all.append(preds)
                y_all.append(batch["y"].numpy())
        return np.concatenate(preds_all), np.concatenate(y_all)

    test_preds, test_gt = predict_loader(test_loader)
    test_acc = (test_preds == test_gt).mean()
    exp_bs["predictions"] = test_preds
    exp_bs["ground_truth"] = test_gt
    exp_bs["test_acc"] = test_acc
    print(f"Test accuracy (bs={bs}): {test_acc:.4f}")

    # ---------- RULE EXTRACTION ----------
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(
        X_train,
        model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy(),
    )
    rule_preds = tree.predict(X_test)
    fidelity = (rule_preds == test_preds).mean()
    exp_bs["rule_preds"] = rule_preds
    exp_bs["fidelity"] = fidelity
    print(f"Rule fidelity (bs={bs}): {fidelity:.4f}")

    # ---------- FAGM ----------
    fagm = np.sqrt(test_acc * fidelity)
    exp_bs["fagm"] = fagm
    print(f"FAGM (bs={bs}): {fagm:.4f}")

    # store
    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = exp_bs

# ---------- SAVE METRICS ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
