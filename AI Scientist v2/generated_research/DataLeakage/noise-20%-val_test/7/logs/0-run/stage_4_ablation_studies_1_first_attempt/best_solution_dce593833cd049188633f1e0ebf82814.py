# Activation-Function Removal (Identity Hidden Layer) Ablation
import os, pathlib, time, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# 0. House-keeping
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

experiment_data = {
    "identity_hidden": {
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "val_rfs": [], "val_loss": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
            "rule_preds": [],
            "test_acc": None,
            "test_rfs": None,
        }
    }
}


# ------------------------------------------------------------------
# 1. Load SPR_BENCH or fallback toy data
# ------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


try:
    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
except Exception as e:
    print("Dataset not found, using synthetic data.", e)
    from datasets import Dataset

    seqs, labels = ["ABAB", "BABA", "AAAA", "BBBB"], [0, 0, 1, 1]
    spr = DatasetDict(
        train=Dataset.from_dict({"sequence": seqs, "label": labels}),
        dev=Dataset.from_dict({"sequence": seqs, "label": labels}),
        test=Dataset.from_dict({"sequence": seqs, "label": labels}),
    )

# ------------------------------------------------------------------
# 2. Vectorise character n-grams
# ------------------------------------------------------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
vectorizer.fit(spr["train"]["sequence"])


def vec(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32)
    y = np.array(split["label"], dtype=np.int64)
    return X, y


X_train, y_train = vec(spr["train"])
X_val, y_val = vec(spr["dev"])
X_test, y_test = vec(spr["test"])
input_dim, num_classes = X_train.shape[1], len(
    set(np.concatenate([y_train, y_val, y_test]))
)


# ------------------------------------------------------------------
# 3. Torch datasets & loaders
# ------------------------------------------------------------------
class CSRTensorDataset(Dataset):
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


train_loader = DataLoader(
    CSRTensorDataset(X_train, y_train), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    CSRTensorDataset(X_val, y_val), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    CSRTensorDataset(X_test, y_test), batch_size=256, shuffle=False, collate_fn=collate
)


# ------------------------------------------------------------------
# 4. Model without activation (identity)
# ------------------------------------------------------------------
class IdentityMLP(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid)
        self.fc2 = nn.Linear(hid, num_classes)

    def forward(self, x):
        return self.fc2(self.fc1(x))  # no non-linearity


# ------------------------------------------------------------------
# 5. Training utils
# ------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, corr, tot = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            out = model(xb)
            l = criterion(out, yb)
            total_loss += l.item() * yb.size(0)
            corr += (out.argmax(1) == yb).sum().item()
            tot += yb.size(0)
    return total_loss / tot, corr / tot


# ------------------------------------------------------------------
# 6. Grid search over hidden dims & L1
# ------------------------------------------------------------------
grid = [(128, 0.0), (256, 1e-4), (256, 1e-3), (512, 1e-4)]
EPOCHS = 8
best_state, best_val, best_cfg = None, -1, None

for hid, l1_coef in grid:
    print(f"\n=== IdentityMLP hid={hid} l1={l1_coef} ===")
    model = IdentityMLP(hid).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss, corr, tot = 0.0, 0, 0
        for batch in train_loader:
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            l1_pen = l1_coef * model.fc1.weight.abs().mean()
            total = loss + l1_pen
            total.backward()
            optim.step()
            run_loss += loss.item() * yb.size(0)
            corr += (out.argmax(1) == yb).sum().item()
            tot += yb.size(0)
        train_acc = corr / tot
        train_loss = run_loss / tot
        val_loss, val_acc = evaluate(model, val_loader)

        # Rule Distillation for RFS
        with torch.no_grad():
            train_soft = (
                model(torch.from_numpy(X_train.toarray()).to(device))
                .argmax(1)
                .cpu()
                .numpy()
            )
        tree = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
            X_train, train_soft
        )
        val_net_preds = (
            model(torch.from_numpy(X_val.toarray()).to(device)).argmax(1).cpu().numpy()
        )
        val_rule_preds = tree.predict(X_val)
        val_rfs = (val_net_preds == val_rule_preds).mean()

        # log
        print(
            f"Epoch {ep}: val_loss={val_loss:.4f} val_acc={val_acc:.3f} RFS={val_rfs:.3f}"
        )
        ed = experiment_data["identity_hidden"]["SPR_BENCH"]
        ed["metrics"]["train_acc"].append(train_acc)
        ed["metrics"]["val_acc"].append(val_acc)
        ed["metrics"]["val_rfs"].append(val_rfs)
        ed["metrics"]["val_loss"].append(val_loss)
        ed["losses"]["train"].append(train_loss)

    if val_acc > best_val:
        best_val, best_state, best_cfg = val_acc, model.state_dict(), (hid, l1_coef)
    del model
    torch.cuda.empty_cache()

print(
    f"Best Identity config: hid={best_cfg[0]} l1={best_cfg[1]} (val_acc={best_val:.4f})"
)

# ------------------------------------------------------------------
# 7. Final evaluation on test
# ------------------------------------------------------------------
best_model = IdentityMLP(best_cfg[0]).to(device)
best_model.load_state_dict(best_state)
best_model.eval()


def collect_preds(loader, m):
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(device)
            preds.append(m(xb).argmax(1).cpu().numpy())
            ys.append(batch["y"].numpy())
    return np.concatenate(preds), np.concatenate(ys)


test_preds, test_gt = collect_preds(test_loader, best_model)
test_acc = (test_preds == test_gt).mean()

train_soft = (
    best_model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy()
)
final_tree = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
    X_train, train_soft
)
rule_test_preds = final_tree.predict(X_test)
test_rfs = (rule_test_preds == test_preds).mean()

ed = experiment_data["identity_hidden"]["SPR_BENCH"]
ed["predictions"], ed["ground_truth"], ed["rule_preds"] = (
    test_preds,
    test_gt,
    rule_test_preds,
)
ed["test_acc"], ed["test_rfs"] = test_acc, test_rfs

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nTEST ACCURACY: {test_acc:.4f}   TEST RFS: {test_rfs:.4f}")

# ------------------------------------------------------------------
# 8. Interpretability: top n-grams
# ------------------------------------------------------------------
W = best_model.fc1.weight.detach().cpu().numpy()  # hid x dim
classifier_W = best_model.fc2.weight.detach().cpu().numpy()  # cls x hid
important = classifier_W @ W  # cls x dim
feature_names = np.array(vectorizer.get_feature_names_out())
topk = 8
for c in range(num_classes):
    idx = np.argsort(-important[c])[:topk]
    feats = ", ".join(feature_names[idx])
    print(f"Class {c} top n-grams: {feats}")
