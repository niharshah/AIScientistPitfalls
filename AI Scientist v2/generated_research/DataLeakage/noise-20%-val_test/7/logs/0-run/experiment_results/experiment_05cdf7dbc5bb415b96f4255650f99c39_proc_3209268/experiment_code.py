import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from datasets import load_dataset, DatasetDict

# ----------------- working dir & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- experiment store -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": [], "RFS": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
        "rule_predictions": [],
    }
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------- data loading -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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
    ds = load_spr_bench(DATA_PATH)
except Exception:
    print("SPR_BENCH not found; using tiny synthetic data.")
    seqs, labels = ["ABAB", "BABA", "AAAA", "BBBB", "AABB", "BBAA"], [0, 0, 1, 1, 0, 1]
    from datasets import Dataset

    tiny = Dataset.from_dict({"sequence": seqs, "label": labels})
    ds = DatasetDict(train=tiny, dev=tiny, test=tiny)

vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
vectorizer.fit(ds["train"]["sequence"])


def vec(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32)
    y = np.array(split["label"], dtype=np.int64)
    return X, y


X_train, y_train = vec(ds["train"])
X_val, y_val = vec(ds["dev"])
X_test, y_test = vec(ds["test"])
input_dim, num_classes = X_train.shape[1], len(
    set(np.concatenate([y_train, y_val, y_test]).tolist())
)
print(f"Loaded data   n_train={len(y_train)}   input_dim={input_dim}")


# ----------------- torch dataset -----------------
class SparseBOWDataset(Dataset):
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
    xb = torch.stack([b["x"] for b in batch])
    yb = torch.stack([b["y"] for b in batch])
    return {"x": xb, "y": yb}


train_loader = DataLoader(
    SparseBOWDataset(X_train, y_train), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SparseBOWDataset(X_val, y_val), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SparseBOWDataset(X_test, y_test), batch_size=256, shuffle=False, collate_fn=collate
)


# ----------------- model -----------------
class MLP(nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(x)


hidden_dim = 256
model = MLP(input_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
l1_lambda = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------- helper funcs -----------------
def evaluate(loader):
    model.eval()
    loss_tot, correct, total = 0.0, 0, 0
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            loss_tot += loss.item() * batch["y"].size(0)
            p = out.argmax(1)
            correct += (p == batch["y"]).sum().item()
            total += batch["y"].size(0)
            ys.append(batch["y"].cpu().numpy())
            preds.append(p.cpu().numpy())
    return loss_tot / total, correct / total, np.concatenate(preds), np.concatenate(ys)


def predict_numpy(loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ----------------- training loop -----------------
EPOCHS = 8
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, tot = 0.0, 0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["x"])
        ce = criterion(out, batch["y"])
        l1 = sum(p.abs().sum() for p in model.parameters())
        loss = ce + l1_lambda * l1
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["y"].size(0)
        correct += (out.argmax(1) == batch["y"]).sum().item()
        tot += batch["y"].size(0)
    train_loss = running_loss / tot
    train_acc = correct / tot

    val_loss, val_acc, val_preds, _ = evaluate(val_loader)

    # ----- rule fidelity -----
    model_preds_train = predict_numpy(train_loader)
    tree = DecisionTreeClassifier(max_depth=5, random_state=SEED)
    tree.fit(X_train, model_preds_train)
    rule_val_preds = tree.predict(X_val)
    RFS = (rule_val_preds == val_preds).mean()

    # store & print
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["RFS"].append(RFS)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  RFS={RFS:.3f}"
    )

# ----------------- final evaluation -----------------
test_preds = predict_numpy(test_loader)
test_acc = (test_preds == y_test).mean()
print(f"\nTest accuracy: {test_acc:.4f}")

# final rule fidelity on test
tree_final = DecisionTreeClassifier(max_depth=5, random_state=SEED)
tree_final.fit(X_train, predict_numpy(train_loader))
rule_test_preds = tree_final.predict(X_test)
test_RFS = (rule_test_preds == test_preds).mean()
print(f"Final Rule Fidelity Score (test): {test_RFS:.4f}")

# save to experiment_data
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = y_test
experiment_data["SPR_BENCH"]["rule_predictions"] = rule_test_preds
experiment_data["SPR_BENCH"]["metrics"]["test_acc"] = test_acc
experiment_data["SPR_BENCH"]["metrics"]["test_RFS"] = test_RFS

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
