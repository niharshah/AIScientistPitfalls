import os, time, random, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict, Dataset

# ------------------------------------------------------------------
# 0. House-keeping & deterministic setup
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {"train_subsample": {}}


# ------------------------------------------------------------------
# 1. Load SPR-BENCH (or tiny fallback)
# ------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


try:
    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
except Exception:
    print("Dataset not found – using tiny synthetic data.")
    seqs, labels = ["ABAB", "BABA", "AAAA", "BBBB"], [0, 0, 1, 1]
    tiny = Dataset.from_dict({"sequence": seqs, "label": labels})
    spr = DatasetDict(train=tiny, dev=tiny, test=tiny)

# ------------------------------------------------------------------
# 2. Text vectoriser
# ------------------------------------------------------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
vectorizer.fit(spr["train"]["sequence"])


def vec(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32)
    y = np.asarray(split["label"], dtype=np.int64)
    return X, y


X_val, y_val = vec(spr["dev"])
X_test, y_test = vec(spr["test"])
input_dim = vectorizer.transform(["tmp"]).shape[1]
num_classes = int(max(spr["train"]["label"])) + 1


# ------------------------------------------------------------------
# 3. Torch dataset wrapper (BUGFIX: inherits from TorchDataset)
# ------------------------------------------------------------------
class CSRTensorDataset(TorchDataset):
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
    x = torch.stack([b["x"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    return {"x": x, "y": y}


# ------------------------------------------------------------------
# 4. Simple MLP
# ------------------------------------------------------------------
class SparseMLP(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.fc1, self.act, self.fc2 = (
            nn.Linear(input_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, num_classes),
        )

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


criterion = nn.CrossEntropyLoss()
grid = [(128, 0.0), (256, 1e-4), (256, 1e-3), (512, 1e-4)]
EPOCHS = 8
SUBSAMPLE_FRACS = [0.10, 0.25, 0.50, 1.00]

full_X_train, full_y_train = vec(spr["train"])
full_size = full_X_train.shape[0]
rng = np.random.RandomState(SEED)

# ------------------------------------------------------------------
# 5. Main ablation loop
# ------------------------------------------------------------------
for frac in SUBSAMPLE_FRACS:
    tag = f"{int(frac*100)}pct"
    experiment_data["train_subsample"][tag] = {
        "metrics": {"train_acc": [], "val_acc": [], "val_rca": [], "val_loss": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": y_test,
        "rule_preds": [],
        "test_acc": None,
        "test_rca": None,
    }

    k = max(1, int(full_size * frac))
    idx = rng.choice(full_size, size=k, replace=False)
    X_train, y_train = full_X_train[idx], full_y_train[idx]

    train_ds = CSRTensorDataset(X_train, y_train)
    val_ds = CSRTensorDataset(X_val, y_val)
    test_ds = CSRTensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

    best_state, best_val, best_cfg = None, -1, None

    for hid, l1_coef in grid:
        model = SparseMLP(hid).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, EPOCHS + 1):
            # ---- training ----
            model.train()
            run_loss, corr, tot = 0.0, 0, 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optim.zero_grad()
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                l1_penalty = l1_coef * model.fc1.weight.abs().mean()
                total_loss = loss + l1_penalty
                total_loss.backward()
                optim.step()

                run_loss += loss.item() * batch["y"].size(0)
                corr += (out.argmax(1) == batch["y"]).sum().item()
                tot += batch["y"].size(0)

            tr_loss, tr_acc = run_loss / tot, corr / tot

            # ---- validation ----
            model.eval()
            val_loss, val_acc, val_tot = 0.0, 0, 0
            net_preds, labels = [], []
            with torch.no_grad():
                for b in val_loader:
                    b = {k: v.to(device) for k, v in b.items()}
                    ob = model(b["x"])
                    l = criterion(ob, b["y"])
                    val_loss += l.item() * b["y"].size(0)
                    val_acc += (ob.argmax(1) == b["y"]).sum().item()
                    val_tot += b["y"].size(0)
                    net_preds.append(ob.argmax(1).cpu().numpy())
                    labels.append(b["y"].cpu().numpy())
            val_loss /= val_tot
            val_acc /= val_tot
            net_preds = np.concatenate(net_preds)
            labels = np.concatenate(labels)

            # ---- rule consistency ----
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
            val_rule_preds = tree.predict(X_val)
            val_rca = ((net_preds == val_rule_preds) & (net_preds == labels)).mean()

            log = experiment_data["train_subsample"][tag]
            log["metrics"]["train_acc"].append(tr_acc)
            log["metrics"]["val_acc"].append(val_acc)
            log["metrics"]["val_rca"].append(val_rca)
            log["metrics"]["val_loss"].append(val_loss)
            log["losses"]["train"].append(tr_loss)

            print(
                f"[{tag}] hid={hid} l1={l1_coef:.0e} epoch={epoch} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_rca={val_rca:.3f}"
            )

        if val_acc > best_val:
            best_val, best_state, best_cfg = val_acc, model.state_dict(), (hid, l1_coef)
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 6. Final evaluation on test split
    # ------------------------------------------------------------------
    best_model = SparseMLP(best_cfg[0]).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    def collect(loader, mdl):
        preds, ys = [], []
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                preds.append(mdl(b["x"]).argmax(1).cpu().numpy())
                ys.append(b["y"].cpu().numpy())
        return np.concatenate(preds), np.concatenate(ys)

    test_preds, ys = collect(test_loader, best_model)
    test_acc = (test_preds == ys).mean()

    train_soft = (
        best_model(torch.from_numpy(X_train.toarray()).to(device))
        .argmax(1)
        .cpu()
        .numpy()
    )
    tree = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
        X_train, train_soft
    )
    rule_test_preds = tree.predict(X_test)
    test_rca = ((test_preds == rule_test_preds) & (test_preds == y_test)).mean()

    log = experiment_data["train_subsample"][tag]
    log["predictions"] = test_preds
    log["rule_preds"] = rule_test_preds
    log["test_acc"] = float(test_acc)
    log["test_rca"] = float(test_rca)

    print(
        f"[{tag}] BEST hid={best_cfg[0]} l1={best_cfg[1]:.0e} "
        f"TEST_ACC={test_acc:.4f} TEST_RCA={test_rca:.4f}"
    )

# ------------------------------------------------------------------
# 7. Persist results
# ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
