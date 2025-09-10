import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
# 0. Set-up & bookkeeping
# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# unified experiment dictionary following required schema
def fresh_ds_dict():
    return dict(
        metrics=dict(train_acc=[], val_acc=[], val_rfs=[], val_loss=[]),
        losses=dict(train=[]),
        predictions=[],
        ground_truth=[],
        rule_preds=[],
        test_acc=None,
        test_rfs=None,
    )


experiment_data = {
    "No_Reg": {"SPR_BENCH": fresh_ds_dict()},
    "L1": {"SPR_BENCH": fresh_ds_dict()},
    "L2": {"SPR_BENCH": fresh_ds_dict()},
}


# -------------------------------------------------
# 1. Load SPR-BENCH (or toy fallback)
# -------------------------------------------------
def load_spr_bench(path: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(path / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


try:
    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
except Exception as e:
    print("Could not load SPR_BENCH, using tiny synthetic toy data.", e)
    from datasets import Dataset

    seqs, labels = ["ABAB", "BABA", "AAAA", "BBBB"], [0, 0, 1, 1]
    ds = Dataset.from_dict({"sequence": seqs, "label": labels})
    spr = DatasetDict(train=ds, dev=ds, test=ds)

# -------------------------------------------------
# 2. Vectorise character n-grams
# -------------------------------------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
vectorizer.fit(spr["train"]["sequence"])


def vec(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32)
    y = np.array(split["label"], dtype=np.int64)
    return X, y


X_train, y_train = vec(spr["train"])
X_val, y_val = vec(spr["dev"])
X_test, y_test = vec(spr["test"])

input_dim = X_train.shape[1]
num_classes = len(set(np.concatenate([y_train, y_val, y_test])))


# -------------------------------------------------
# 3. Torch datasets / loaders
# -------------------------------------------------
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


# -------------------------------------------------
# 4. Model definition
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid, num_classes)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# -------------------------------------------------
# 5. Helper functions
# -------------------------------------------------
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        xb, yb = batch["x"].to(device), batch["y"].to(device)
        out = model(xb)
        l = criterion(out, yb)
        tot_loss += l.item() * yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return tot_loss / total, correct / total


def rule_fidelity(model, X_val):
    # Distill decision tree on train soft labels then measure agreement on val
    train_soft = (
        model(torch.from_numpy(X_train.toarray()).to(device)).argmax(1).cpu().numpy()
    )
    tree = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
        X_train, train_soft
    )
    val_net = (
        model(torch.from_numpy(X_val.toarray()).to(device)).argmax(1).cpu().numpy()
    )
    val_tree = tree.predict(X_val)
    return (val_net == val_tree).mean()


# -------------------------------------------------
# 6. Ablation loop
# -------------------------------------------------
EPOCHS = 8
ablations = ["No_Reg", "L1", "L2"]

for abl in ablations:
    print("\n==========================")
    print(f"Running ablation: {abl}")
    print("==========================")

    # grid: (hidden_dim, reg_strength) where reg_strength
    # = l1_coef for L1, weight_decay for L2, ignored for No_Reg
    grid = [(128, 0.0), (256, 1e-4), (256, 1e-3), (512, 1e-4)]

    best_state, best_val_acc, best_cfg = None, -1, None

    for hid, reg_strength in grid:
        print(f"\n--- cfg: hid={hid}, reg={reg_strength} ---")
        model = MLP(hid).to(device)
        weight_decay = 0.0
        l1_coef = 0.0
        if abl == "L2":
            weight_decay = reg_strength
        elif abl == "L1":
            l1_coef = reg_strength

        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            run_loss, corr, tot = 0.0, 0, 0
            for batch in train_loader:
                xb, yb = batch["x"].to(device), batch["y"].to(device)
                optim.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                if l1_coef > 0:
                    loss = loss + l1_coef * model.fc1.weight.abs().mean()
                loss.backward()
                optim.step()

                run_loss += loss.item() * yb.size(0)
                corr += (out.argmax(1) == yb).sum().item()
                tot += yb.size(0)

            train_acc = corr / tot
            train_loss = run_loss / tot
            val_loss, val_acc = evaluate(model, val_loader)
            val_rfs = rule_fidelity(model, X_val)

            # logging
            edict = experiment_data[abl]["SPR_BENCH"]
            edict["metrics"]["train_acc"].append(train_acc)
            edict["metrics"]["val_acc"].append(val_acc)
            edict["metrics"]["val_rfs"].append(val_rfs)
            edict["metrics"]["val_loss"].append(val_loss)
            edict["losses"]["train"].append(train_loss)

            print(
                f"Epoch {epoch}: train_acc={train_acc:.3f} "
                f"val_acc={val_acc:.3f} RFS={val_rfs:.3f}"
            )

        # keep best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            best_cfg = (hid, reg_strength)
        del model
        torch.cuda.empty_cache()

    print(f"Best cfg for {abl}: hid={best_cfg[0]} reg_strength={best_cfg[1]}")

    # -------------------------------------------------
    # 7. Final evaluation on test
    # -------------------------------------------------
    best_model = MLP(best_cfg[0]).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    @torch.no_grad()
    def collect_preds(loader, model):
        preds, ys = [], []
        for batch in loader:
            xb = batch["x"].to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
            ys.append(batch["y"].numpy())
        return np.concatenate(preds), np.concatenate(ys)

    test_preds, test_gt = collect_preds(test_loader, best_model)
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

    edict = experiment_data[abl]["SPR_BENCH"]
    edict["predictions"] = test_preds
    edict["ground_truth"] = test_gt
    edict["rule_preds"] = rule_test_preds
    edict["test_acc"] = test_acc
    edict["test_rfs"] = test_rfs

    print(f"{abl} --> TEST ACC: {test_acc:.4f}   TEST RFS: {test_rfs:.4f}")

    # -------------------------------------------------
    # 8. Simple interpretability: top n-grams per class
    # -------------------------------------------------
    W1 = best_model.fc1.weight.detach().cpu().numpy()  # hid x dim
    Wc = best_model.fc2.weight.detach().cpu().numpy()  # cls x hid
    important = Wc @ W1  # cls x dim
    feature_names = np.array(vectorizer.get_feature_names_out())
    topk = 8
    for c in range(num_classes):
        idx = np.argsort(-important[c])[:topk]
        feats = ", ".join(feature_names[idx])
        print(f"[{abl}] Class {c} top n-grams: {feats}")

# -------------------------------------------------
# 9. Save everything
# -------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll results saved to", os.path.join(working_dir, "experiment_data.npy"))
