import os, pathlib, random, time, numpy as np, torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import disable_caching

disable_caching()

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment store ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "train_cpx": [], "val_cpx": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- helper metrics ----------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def complexity_weight(seq):  # product of varieties
    return count_color_variety(seq) * count_shape_variety(seq)


def cpx_weighted_accuracy(seqs, y_true, y_pred):
    w = np.array([complexity_weight(s) for s in seqs])
    correct = (y_true == y_pred).astype(int)
    return (w * correct).sum() / (w.sum() + 1e-9)


# ---------- load dataset ----------
DATA_PATH_CAND = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
data_path = next((p for p in DATA_PATH_CAND if p.exists()), None)
assert data_path is not None, "SPR_BENCH folder not found."

# Re-use loader code from prompt
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset("csv", data_files=str(root / split_csv), split="train")

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


spr = load_spr_bench(data_path)
print({k: len(v) for k, v in spr.items()})

# ---------- build glyph vocabulary ----------
all_tokens = []
for seq in spr["train"]["sequence"]:
    all_tokens.extend(seq.split())


# numeric encoding: 2 dims (ascii of chars)
def glyph_to_vec(g):
    return [ord(g[0]) / 128.0, ord(g[1]) / 128.0]  # scale roughly


glyph_vecs = np.array([glyph_to_vec(g) for g in all_tokens])
k = 8
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
km.fit(glyph_vecs)
print("KMeans trained.")


# ---------- vectorise sequences ----------
def seq_to_hist(seq):
    hist = np.zeros(k, dtype=np.float32)
    for tok in seq.split():
        cid = km.predict([glyph_to_vec(tok)])[0]
        hist[cid] += 1.0
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def vectorise_split(split):
    X, y, seqs = [], [], []
    for ex in spr[split]:
        X.append(seq_to_hist(ex["sequence"]))
        y.append(int(ex["label"]))
        seqs.append(ex["sequence"])
    return np.stack(X), np.array(y, dtype=np.int64), seqs


X_train, y_train, seq_train = vectorise_split("train")
X_dev, y_dev, seq_dev = vectorise_split("dev")
X_test, y_test, seq_test = vectorise_split("test")


# ---------- Dataset wrappers ----------
class HistSet(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


batch_size = 128
train_loader = DataLoader(
    HistSet(X_train, y_train), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(HistSet(X_dev, y_dev), batch_size=batch_size)


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, inp, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


n_classes = int(max(y_train.max(), y_dev.max(), y_test.max()) + 1)
model = MLP(k, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- training loop ----------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        out = model(batch["x"])
        loss = criterion(out, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["y"].size(0)
        preds = out.argmax(1)
        correct += (preds == batch["y"]).sum().item()
        total += batch["y"].size(0)
    train_loss = total_loss / total
    train_acc = correct / total

    # validation
    model.eval()
    val_loss, v_correct, v_total = 0.0, 0, 0
    all_val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            val_loss += loss.item() * batch["y"].size(0)
            preds = out.argmax(1)
            all_val_preds.extend(preds.cpu().numpy())
            v_correct += (preds == batch["y"]).sum().item()
            v_total += batch["y"].size(0)
    val_loss /= v_total
    val_acc = v_correct / v_total
    train_cpx = cpx_weighted_accuracy(
        seq_train,
        y_train,
        model(torch.tensor(X_train, device=device)).argmax(1).cpu().numpy(),
    )
    val_cpx = cpx_weighted_accuracy(seq_dev, y_dev, np.array(all_val_preds))

    # store
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["train_cpx"].append(train_cpx)
    experiment_data["SPR_BENCH"]["metrics"]["val_cpx"].append(val_cpx)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.3f} | val_CpxWA={val_cpx:.3f}"
    )

# ---------- final test evaluation ----------
model.eval()
test_preds = model(torch.tensor(X_test, device=device)).argmax(1).cpu().numpy()
test_acc = (test_preds == y_test).mean()
test_cpx = cpx_weighted_accuracy(seq_test, y_test, test_preds)
print(f"\nTEST  PlainAcc={test_acc:.3f}  CpxWA={test_cpx:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = y_test
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
