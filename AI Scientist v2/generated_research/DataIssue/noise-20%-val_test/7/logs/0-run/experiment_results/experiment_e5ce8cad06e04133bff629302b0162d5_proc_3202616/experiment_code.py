# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# ----- 0. Imports & Repro -----
import os, pathlib, time, numpy as np, torch, torch.nn as nn, random
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from typing import Dict
from datasets import load_dataset, DatasetDict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----- 1. House-keeping -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# unified experiment store
experiment_data: Dict = {
    "hidden_dim": {  # hyperparam tuning type
        "SPR_BENCH": {
            "metrics": {},  # filled per hidden_dim
            "losses": {},
            "predictions": [],  # best model
            "ground_truth": [],
            "rule_preds": [],
            "best_hidden_dim": None,
        }
    }
}


# ----- 2. Data Loading -----
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


DATA_PATH = pathlib.Path("./SPR_BENCH")
try:
    spr = load_spr_bench(DATA_PATH)
except FileNotFoundError:
    print("SPR_BENCH not found, creating synthetic toy data.")
    seqs, labels = ["ABAB", "BABA", "AAAA", "BBBB"], [0, 0, 1, 1]
    from datasets import Dataset

    dsmall = Dataset.from_dict({"sequence": seqs, "label": labels})
    spr = DatasetDict(train=dsmall, dev=dsmall, test=dsmall)

# ----- 3. Vectorisation -----
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
vectorizer.fit(spr["train"]["sequence"])


def vec(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32)
    y = np.array(split["label"], dtype=np.int64)
    return X, y


X_train, y_train = vec(spr["train"])
X_val, y_val = vec(spr["dev"])
X_test, y_test = vec(spr["test"])
input_dim, num_classes = X_train.shape[1], len(
    set(np.concatenate([y_train, y_val, y_test]).tolist())
)
print(f"input_dim={input_dim}  num_classes={num_classes}")


# ----- 4. Dataset & Loader -----
class CSRDataset(Dataset):
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
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


train_ds, val_ds, test_ds = (
    CSRDataset(X_train, y_train),
    CSRDataset(X_val, y_val),
    CSRDataset(X_test, y_test),
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

# ----- 5. Hyper-parameter grid search -----
hidden_dims = [32, 64, 128, 256, 512]
EPOCHS = 5
best_dim, best_val_acc = None, -1
best_state_dict = None

for hd in hidden_dims:
    print(f"\n===== Hidden dim {hd} =====")
    # model, loss, optim
    model = nn.Sequential(
        nn.Linear(input_dim, hd), nn.ReLU(), nn.Linear(hd, num_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # per-dim stores
    train_acc_l, val_acc_l, val_loss_l, train_loss_l = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        # val
        model.eval()
        vloss, vcorr, vtot = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch["x"].to(device), batch["y"].to(device)
                out = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item() * yb.size(0)
                vcorr += (out.argmax(1) == yb).sum().item()
                vtot += yb.size(0)
        vloss /= vtot
        val_acc = vcorr / vtot
        # log
        train_loss_l.append(train_loss)
        train_acc_l.append(train_acc)
        val_loss_l.append(vloss)
        val_acc_l.append(val_acc)
        print(f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
    # save per dim
    experiment_data["hidden_dim"]["SPR_BENCH"]["metrics"][hd] = {
        "train_acc": train_acc_l,
        "val_acc": val_acc_l,
        "val_loss": val_loss_l,
    }
    experiment_data["hidden_dim"]["SPR_BENCH"]["losses"][hd] = {"train": train_loss_l}
    # choose best
    if val_acc_l[-1] > best_val_acc:
        best_val_acc = val_acc_l[-1]
        best_dim = hd
        best_state_dict = model.state_dict()
    # free memory
    del model
    torch.cuda.empty_cache()

print(f"\nBest hidden_dim = {best_dim} (val_acc={best_val_acc:.4f})")
experiment_data["hidden_dim"]["SPR_BENCH"]["best_hidden_dim"] = best_dim

# ----- 6. Re-instantiate best model for final evaluation -----
best_model = nn.Sequential(
    nn.Linear(input_dim, best_dim), nn.ReLU(), nn.Linear(best_dim, num_classes)
).to(device)
best_model.load_state_dict(best_state_dict)
best_model.eval()


def predict(loader, m):
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            preds.append(m(x).argmax(1).cpu().numpy())
            ys.append(batch["y"].numpy())
    return np.concatenate(preds), np.concatenate(ys)


test_preds, test_gt = predict(test_loader, best_model)
test_acc = (test_preds == test_gt).mean()
print(f"Test accuracy (best model): {test_acc:.4f}")

# ----- 7. Rule extraction -----
tree = DecisionTreeClassifier(max_depth=5, random_state=SEED)
train_soft = (
    best_model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy()
)
tree.fit(X_train, train_soft)
rule_preds = tree.predict(X_test)
fidelity = (rule_preds == test_preds).mean()
print(f"Rule fidelity: {fidelity:.4f}   FAGM={np.sqrt(test_acc*fidelity):.4f}")

# ----- 8. Save predictions & extras -----
experiment_data["hidden_dim"]["SPR_BENCH"]["predictions"] = test_preds
experiment_data["hidden_dim"]["SPR_BENCH"]["ground_truth"] = test_gt
experiment_data["hidden_dim"]["SPR_BENCH"]["rule_preds"] = rule_preds
experiment_data["hidden_dim"]["SPR_BENCH"]["metrics"]["best_test_acc"] = test_acc
experiment_data["hidden_dim"]["SPR_BENCH"]["metrics"]["best_fidelity"] = fidelity

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
