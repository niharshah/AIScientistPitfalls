import os, random, pathlib, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------ FOLDER SET-UP --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------ DEVICE ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------- REPRODUCIBILITY ----------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ------------------------- DATA LOADING --------------------------------
def load_spr_bench(root) -> DatasetDict:
    """Return HuggingFace DatasetDict for SPR_BENCH splits."""
    root = pathlib.Path(root)  # <-- BUGFIX: ensure Path

    def _load(csv_name: str):
        csv_path = root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected {csv_path} – check SPR_BENCH path")
        return load_dataset(
            "csv", data_files=str(csv_path), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = os.environ.get(
    "SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})

# ----------------------- VOCAB & ENCODING ------------------------------
train_seqs = dsets["train"]["sequence"]
chars = sorted({c for s in train_seqs for c in s})
char2id = {c: i + 1 for i, c in enumerate(chars)}  # 0=PAD
vocab_size = len(char2id) + 1
max_len = max(len(s) for s in train_seqs)
print(f"vocab={vocab_size-1}, max_len={max_len}")

lbls = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(lbls)}
num_classes = len(lbls)


def encode_split(split):
    seqs = dsets[split]["sequence"]
    X = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids = [char2id[c] for c in s][:max_len]
        X[i, : len(ids)] = ids
    y = np.array([lab2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y


X_train, y_train = encode_split("train")
X_dev, y_dev = encode_split("dev")
X_test, y_test = encode_split("test")


class CharDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X[idx]), "y": torch.tensor(self.y[idx])}


batch_size = 256
train_loader = DataLoader(CharDataset(X_train, y_train), batch_size, shuffle=True)
dev_loader = DataLoader(CharDataset(X_dev, y_dev), batch_size)
test_loader = DataLoader(CharDataset(X_test, y_test), batch_size)


# --------------------------- MODEL -------------------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab, embed_dim=32, filters=64, ks=(3, 4, 5), classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in ks])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(filters * len(ks), classes)

    def forward(self, x, return_feat=False):
        x = self.emb(x).transpose(1, 2)  # B,E,T -> B,E,L
        feats = [torch.relu(c(x)).max(-1)[0] for c in self.convs]
        feats = torch.cat(feats, 1)
        out = self.fc(self.dropout(feats))
        return (out, feats) if return_feat else out


model = TextCNN(vocab_size, classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------- EXPERIMENT RECORD -------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "rule_fid": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "timestamps": [],
    }
}


# ------------------- RULE FIDELITY HELPERS -----------------------------
def rule_fidelity(model, loader):
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        import subprocess, sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "scikit-learn", "--quiet"]
        )
        from sklearn.tree import DecisionTreeClassifier
    model.eval()
    feats, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits, f = model(x, return_feat=True)
            feats.append(f.cpu().numpy())
            preds.append(torch.argmax(logits, 1).cpu().numpy())
    feats = np.concatenate(feats)
    preds = np.concatenate(preds)
    dt = DecisionTreeClassifier(max_depth=5, random_state=0).fit(feats, preds)
    fid = (dt.predict(feats) == preds).mean()
    return fid


# ---------------------------- TRAIN & EVAL -----------------------------
def evaluate(loader):
    model.eval()
    tot, cor, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            logits = model(b["x"])
            loss = criterion(logits, b["y"])
            preds = logits.argmax(1)
            tot += b["y"].size(0)
            cor += (preds == b["y"]).sum().item()
            loss_sum += loss.item() * b["y"].size(0)
    return cor / tot, loss_sum / tot


epochs = 8
best_val, best_preds = -1, None
for epoch in range(1, epochs + 1):
    model.train()
    tot, cor, loss_sum = 0, 0, 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        preds = logits.argmax(1)
        tot += batch["y"].size(0)
        cor += (preds == batch["y"]).sum().item()
        loss_sum += loss.item() * batch["y"].size(0)
    tr_acc, tr_loss = cor / tot, loss_sum / tot
    val_acc, val_loss = evaluate(dev_loader)
    rfa = rule_fidelity(model, dev_loader)

    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["rule_fid"].append(rfa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"train_acc={tr_acc:.3f} val_acc={val_acc:.3f} RFA={rfa:.3f}"
    )

    if val_acc > best_val:
        best_val = val_acc
        test_acc, _ = evaluate(test_loader)
        print(f"  ** new best dev_acc; test_acc={test_acc:.3f}")
        # store preds
        model.eval()
        preds = []
        with torch.no_grad():
            for b in test_loader:
                logits = model(b["x"].to(device))
                preds.append(logits.argmax(1).cpu().numpy())
        best_preds = np.concatenate(preds)

# ------------------------------ SAVE -----------------------------------
experiment_data["SPR_BENCH"]["predictions"] = best_preds.tolist()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics & predictions to working/experiment_data.npy")
