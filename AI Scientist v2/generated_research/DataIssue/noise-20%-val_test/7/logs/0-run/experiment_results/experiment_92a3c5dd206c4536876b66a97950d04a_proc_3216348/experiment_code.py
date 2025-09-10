# ---------------------------------------------------------------
# Binary N-gram Feature Ablation: single-file runnable script
# ---------------------------------------------------------------
import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict

# ---------------- House-keeping ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# unified logging dict in requested format
experiment_data = {
    "BinaryNgramFeature": {
        # "count" and "binary" entries will be filled below
    }
}


# ---------------- Data load ----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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
    spr_all = load_spr_bench(DATA_PATH)
except Exception as e:
    # fallback tiny synthetic dataset
    from datasets import Dataset

    seqs, labels = ["ABAB", "BABA", "AAAA", "BBBB", "ABAA", "BAAA", "ABBB", "BBBA"], [
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
    ]
    ds = Dataset.from_dict({"sequence": seqs, "label": labels})
    spr_all = DatasetDict(train=ds, dev=ds, test=ds)


# ---------------- Helper DS ----------------
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


def collate_fn(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


# ---------------- Model ----------------
class SparseMLP(nn.Module):
    def __init__(self, input_dim, hid, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid, num_classes)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


criterion = nn.CrossEntropyLoss()
grid = [(128, 0.0), (256, 1e-4), (256, 1e-3), (512, 1e-4)]
EPOCHS = 8


def evaluate(model, loader):
    model.eval()
    loss_tot, corr, tot = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            out = model(xb)
            loss = criterion(out, yb)
            loss_tot += loss.item() * yb.size(0)
            preds = out.argmax(1)
            corr += (preds == yb).sum().item()
            tot += yb.size(0)
    return loss_tot / tot, corr / tot


# ---------------- Ablation loop ----------------
for mode_name, mode_cfg in {"count": False, "binary": True}.items():
    print(f"\n############## Running mode: {mode_name} #############")
    # Vectoriser
    vectorizer = CountVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=1, binary=mode_cfg
    )
    vectorizer.fit(spr_all["train"]["sequence"])

    def vec(split):
        X = vectorizer.transform(split["sequence"]).astype(np.float32)
        y = np.array(split["label"], dtype=np.int64)
        return X, y

    X_train, y_train = vec(spr_all["train"])
    X_val, y_val = vec(spr_all["dev"])
    X_test, y_test = vec(spr_all["test"])
    input_dim = X_train.shape[1]
    num_classes = len(set(np.concatenate([y_train, y_val, y_test])))

    # Torch datasets/loaders
    train_ds = CSRTensorDataset(X_train, y_train)
    val_ds = CSRTensorDataset(X_val, y_val)
    test_ds = CSRTensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, 128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, 256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, 256, shuffle=False, collate_fn=collate_fn)

    # storage
    exp_entry = {
        "metrics": {"train_acc": [], "val_acc": [], "val_rfs": [], "val_loss": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
        "rule_preds": [],
        "test_acc": None,
        "test_rfs": None,
    }

    # grid search
    best_state, best_val, best_cfg = None, -1, None
    for hid, l1_coef in grid:
        print(f"\n== cfg hid={hid} l1={l1_coef} ==")
        model = SparseMLP(input_dim, hid, num_classes).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, EPOCHS + 1):
            # train
            model.train()
            run_loss, corr, tot = 0.0, 0, 0
            for batch in train_loader:
                xb, yb = batch["x"].to(device), batch["y"].to(device)
                optim.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                l1_pen = l1_coef * model.fc1.weight.abs().mean()
                total_loss = loss + l1_pen
                total_loss.backward()
                optim.step()
                run_loss += loss.item() * yb.size(0)
                corr += (out.argmax(1) == yb).sum().item()
                tot += yb.size(0)
            train_acc = corr / tot
            train_loss = run_loss / tot

            val_loss, val_acc = evaluate(model, val_loader)
            # rule fidelity on val
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
                model(torch.from_numpy(X_val.toarray()).to(device))
                .argmax(1)
                .cpu()
                .numpy()
            )
            val_rule_preds = tree.predict(X_val)
            val_rfs = (val_net_preds == val_rule_preds).mean()

            exp_entry["metrics"]["train_acc"].append(train_acc)
            exp_entry["metrics"]["val_acc"].append(val_acc)
            exp_entry["metrics"]["val_loss"].append(val_loss)
            exp_entry["metrics"]["val_rfs"].append(val_rfs)
            exp_entry["losses"]["train"].append(train_loss)

            print(
                f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} RFS={val_rfs:.3f}"
            )

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
            best_cfg = (hid, l1_coef)
        del model
        torch.cuda.empty_cache()

    print(
        f"Best (mode {mode_name}): hid={best_cfg[0]} l1={best_cfg[1]} val={best_val:.4f}"
    )

    # --- final evaluation ---
    best_model = SparseMLP(input_dim, best_cfg[0], num_classes).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    def collect(loader, mdl):
        preds, ys = [], []
        with torch.no_grad():
            for batch in loader:
                xb = batch["x"].to(device)
                preds.append(mdl(xb).argmax(1).cpu().numpy())
                ys.append(batch["y"].numpy())
        return np.concatenate(preds), np.concatenate(ys)

    test_preds, test_gt = collect(test_loader, best_model)
    test_acc = (test_preds == test_gt).mean()

    # rule fidelity on test
    train_soft = (
        best_model(torch.from_numpy(X_train.toarray()).to(device))
        .argmax(1)
        .cpu()
        .numpy()
    )
    final_tree = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
        X_train, train_soft
    )
    rule_test_preds = final_tree.predict(X_test)
    test_rfs = (rule_test_preds == test_preds).mean()

    exp_entry["predictions"] = test_preds
    exp_entry["ground_truth"] = test_gt
    exp_entry["rule_preds"] = rule_test_preds
    exp_entry["test_acc"] = test_acc
    exp_entry["test_rfs"] = test_rfs

    # print summary
    print(f"TEST ACCURACY ({mode_name}): {test_acc:.4f}   TEST RFS: {test_rfs:.4f}")

    # interpretability: print top features
    W1 = best_model.fc1.weight.detach().cpu().numpy()
    W2 = best_model.fc2.weight.detach().cpu().numpy()
    importance = W2 @ W1
    feature_names = np.array(vectorizer.get_feature_names_out())
    topk = min(8, feature_names.size)
    for c in range(num_classes):
        idx = np.argsort(-importance[c])[:topk]
        print(f"Class {c} top n-grams ({mode_name}): {', '.join(feature_names[idx])}")

    # store into experiment_data dict
    experiment_data["BinaryNgramFeature"][mode_name] = exp_entry

# ---------------- Save ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll experiments finished & saved to experiment_data.npy")
