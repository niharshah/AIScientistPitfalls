import os, pathlib, time, random, numpy as np, torch, torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader

# ----------------------- house-keeping -----------------------------
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


# ----------------------- data loading ------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


try:
    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
except Exception as e:
    print("Real dataset not found, using toy synthetic.", e)
    from datasets import Dataset

    toyX, toyy = ["ABAB", "BABA", "AAAA", "BBBB", "AABB", "BBAA"], [0, 0, 1, 1, 0, 1]
    spr = DatasetDict(
        train=Dataset.from_dict({"sequence": toyX, "label": toyy}),
        dev=Dataset.from_dict({"sequence": toyX, "label": toyy}),
        test=Dataset.from_dict({"sequence": toyX, "label": toyy}),
    )

# ------------------- vectorisation & binarisation ------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, binary=True)
vectorizer.fit(spr["train"]["sequence"])


def vec(split):
    X = vectorizer.transform(split["sequence"]).astype(np.float32).todense()
    y = np.asarray(split["label"], dtype=np.int64)
    return torch.tensor(X), torch.tensor(y)


X_tr, y_tr = vec(spr["train"])
X_v, y_v = vec(spr["dev"])
X_te, y_te = vec(spr["test"])
input_dim, num_classes = X_tr.shape[1], int(max(y_tr.max(), y_v.max(), y_te.max())) + 1
feat_names = np.array(vectorizer.get_feature_names_out())


class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
v_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=512, shuffle=False)
te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=512, shuffle=False)


# ------------------ model : soft-AND rule layer --------------------
class NeuralRuleLayer(nn.Module):
    def __init__(self, in_dim, n_rules):
        super().__init__()
        self.gates = nn.Parameter(torch.randn(n_rules, in_dim))  # real → sigmoid
        self.temp = 5.0  # annealed

    def forward(self, x):
        g = torch.sigmoid(self.gates * self.temp)  # (R,D)
        # soft AND : ∏ (g_i*x + 1-g_i)  == 1 - ∏ (1-g_i*x)
        # here use log-sum to avoid underflow
        prod = torch.exp((torch.log(g.unsqueeze(0) * x.unsqueeze(1) + 1e-8)).sum(-1))
        return prod  # (B,R)

    def extract_rules(self, thr=0.9):
        hard = torch.sigmoid(self.gates * self.temp).cpu().detach().numpy() > thr
        rules = []
        for r, mask in enumerate(hard):
            feats = feat_names[mask]
            if len(feats):
                rules.append((r, feats))
        return rules


class InterpretableNet(nn.Module):
    def __init__(self, in_dim, n_rules, classes):
        super().__init__()
        self.rules = NeuralRuleLayer(in_dim, n_rules)
        self.classifier = nn.Linear(n_rules, classes)

    def forward(self, x):
        h = self.rules(x)
        return self.classifier(h)


# ---------------- training utilities ------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    loss = corr = tot = 0
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            out = model(xb)
            l = criterion(out, yb)
            loss += l.item() * yb.size(0)
            pred = out.argmax(1)
            corr += (pred == yb).sum().item()
            tot += yb.size(0)
    return loss / tot, corr / tot


def collect_preds(model, loader):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in loader:
            xb = b["x"].to(device)
            P.append(model(xb).argmax(1).cpu())
            Y.append(b["y"])
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


# --------------------- training loop ------------------------------
model = InterpretableNet(input_dim, n_rules=128, classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, steps_per_epoch=len(tr_loader), epochs=20
)
l1_coef = 1e-4
best_val = -1
patience = 4
bad_epochs = 0
best_state = None
for epoch in range(1, 21):
    model.train()
    run_loss = corr = tot = 0
    for batch in tr_loader:
        xb = batch["x"].to(device)
        yb = batch["y"].to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        # L1 on rule gates to promote sparsity
        l1 = model.rules.gates.abs().mean() * l1_coef
        total = loss + l1
        total.backward()
        optimizer.step()
        scheduler.step()
        run_loss += loss.item() * yb.size(0)
        corr += (out.argmax(1) == yb).sum().item()
        tot += yb.size(0)
    tr_acc = corr / tot
    tr_loss = run_loss / tot

    val_loss, val_acc = evaluate(model, v_loader, criterion)

    # ----- rule fidelity via decision tree -----
    tr_net_pred = collect_preds(model, tr_loader)[0]
    DT = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
        X_tr.numpy(), tr_net_pred
    )
    v_net, _ = collect_preds(model, v_loader)
    v_rule_pred = DT.predict(X_v.numpy())
    val_rfs = (v_net == v_rule_pred).mean()

    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.3f} RFS={val_rfs:.3f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_rfs"].append(val_rfs)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    if val_acc > best_val:
        best_val = val_acc
        best_state = model.state_dict()
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs >= patience:
        print("Early stop.")
        break
    # anneal temperature for crisper rules
    model.rules.temp = max(1.0, model.rules.temp * 0.9)

# ---------------------- test evaluation ---------------------------
model.load_state_dict(best_state)
test_preds, test_gt = collect_preds(model, te_loader)
test_acc = (test_preds == test_gt).mean()
DT_final = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
    X_tr.numpy(), collect_preds(model, tr_loader)[0]
)
rule_test_preds = DT_final.predict(X_te.numpy())
test_rfs = (rule_test_preds == test_preds).mean()

experiment_data["SPR_BENCH"]["predictions"] = test_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt.tolist()
experiment_data["SPR_BENCH"]["rule_preds"] = rule_test_preds.tolist()
experiment_data["SPR_BENCH"]["test_acc"] = float(test_acc)
experiment_data["SPR_BENCH"]["test_rfs"] = float(test_rfs)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nTEST ACCURACY: {test_acc:.4f}  TEST RFS: {test_rfs:.4f}")

# --------------------- print extracted rules ----------------------
for r, feats in model.rules.extract_rules():
    print(
        f"Rule {r}: IF sequence contains {{", ", ".join(feats[:6]), "}} THEN hidden=1"
    )
