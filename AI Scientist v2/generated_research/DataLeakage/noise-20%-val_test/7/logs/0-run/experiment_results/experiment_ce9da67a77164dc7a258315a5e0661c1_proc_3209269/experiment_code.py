import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer

# ---------- house-keeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# unified experiment store
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": [], "rfs": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rule_preds": [],
    }
}


# ---------- data ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path("./SPR_BENCH")
try:
    spr = load_spr_bench(DATA_PATH)
except Exception:
    print("Dataset not found – using synthetic toy corpus.")
    from datasets import Dataset

    toy_seq, toy_lab = ["ABAB", "ABBA", "AAAA", "BBBB", "BBAA", "AABB"], [
        0,
        0,
        1,
        1,
        0,
        1,
    ]
    spr = DatasetDict()
    spr["train"] = spr["dev"] = spr["test"] = Dataset.from_dict(
        {"sequence": toy_seq * 300, "label": toy_lab * 300}
    )

# ---------- vectoriser ----------
VEC_N = 3
vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(VEC_N, VEC_N), binary=True, min_df=1
)
vectorizer.fit(spr["train"]["sequence"])


def vec(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32)
    y = np.array(split["label"], dtype=np.int64)
    return X, y


X_train, y_train = vec(spr["train"])
X_val, y_val = vec(spr["dev"])
X_test, y_test = vec(spr["test"])
input_dim, num_classes = X_train.shape[1], int(np.max(y_train)) + 1
print(f"input_dim={input_dim}, classes={num_classes}")


# ---------- dataset/loader ----------
class CSRSplit(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return {
            "x": torch.from_numpy(self.X[i].toarray()).squeeze(0),
            "y": torch.tensor(self.y[i]),
        }


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


train_loader = DataLoader(
    CSRSplit(X_train, y_train), batch_size=256, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    CSRSplit(X_val, y_val), batch_size=512, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    CSRSplit(X_test, y_test), batch_size=512, shuffle=False, collate_fn=collate
)


# ---------- sparse logistic model ----------
class SparseLinear(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc = nn.Linear(inp, out)

    def forward(self, x):
        return self.fc(x)


l1_grid = [1e-4, 5e-4, 1e-3]
best_state, best_val, best_lambda = None, -1, None
EPOCHS = 20


def run_lambda(l1_lambda):
    model = SparseLinear(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        # ------- train -------
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb) + l1_lambda * model.fc.weight.abs().mean()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_loss = tot_loss / total
        train_acc = correct / total
        # ------- val ---------
        model.eval()
        vloss, vcorr, vtot = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch["x"].to(device), batch["y"].to(device)
                logits = model(xb)
                loss = ce(logits, yb) + l1_lambda * model.fc.weight.abs().mean()
                vloss += loss.item() * yb.size(0)
                vcorr += (logits.argmax(1) == yb).sum().item()
                vtot += yb.size(0)
        val_loss, val_acc = vloss / vtot, vcorr / vtot
        # --- log ---
        experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
        experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
        experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        print(
            f"λ={l1_lambda} Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
    return model, val_acc


for lmbd in l1_grid:
    model, val_acc = run_lambda(lmbd)
    if val_acc > best_val:
        best_val, best_state, best_lambda = val_acc, model.state_dict(), lmbd

print(f"Best λ={best_lambda} with dev acc {best_val:.3f}")
model = SparseLinear(input_dim, num_classes).to(device)
model.load_state_dict(best_state)
model.eval()


# ---------- evaluation ----------
def predict(loader, m):
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(device)
            preds.append(m(xb).argmax(1).cpu().numpy())
            ys.append(batch["y"].numpy())
    return np.concatenate(preds), np.concatenate(ys)


test_pred, test_gt = predict(test_loader, model)
test_acc = (test_pred == test_gt).mean()
print(f"Test accuracy: {test_acc:.3f}")

# ---------- rule extraction ----------
weights = model.fc.weight.detach().cpu().numpy()  # shape C x D
k = 10
feature_names = np.array(vectorizer.get_feature_names_out())
rule_preds = []
for row in X_test:
    feats = row.indices
    scores = np.zeros(num_classes)
    for c in range(num_classes):
        top_idx = np.argsort(-np.abs(weights[c]))[:k]
        inter = np.intersect1d(feats, top_idx, assume_unique=False)
        scores[c] = weights[c, inter].sum()
    rule_preds.append(np.argmax(scores))
rule_preds = np.array(rule_preds)
rfs = (rule_preds == test_pred).mean()
print(f"Rule Fidelity Score (top-{k} features): {rfs:.3f}")
experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt
experiment_data["SPR_BENCH"]["rule_preds"] = rule_preds
experiment_data["SPR_BENCH"]["metrics"]["rfs"].append(rfs)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
