import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------- working dir & device -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- reproducibility ---------------------------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ---------------- SPR loader --------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ---------------- vectoriser (uni+bi grams) -----------------------------------
def build_vocab(seqs):
    unis, bis = set(), set()
    for s in seqs:
        unis.update(s)
        bis.update([s[i : i + 2] for i in range(len(s) - 1)])
    vocab = sorted(unis) + sorted(bis)
    return {tok: i for i, tok in enumerate(vocab)}


def vectorise(s, idx):
    v = np.zeros(len(idx), dtype=np.float32)
    for c in s:
        if c in idx:
            v[idx[c]] += 1.0
    for i in range(len(s) - 1):
        bg = s[i : i + 2]
        if bg in idx:
            v[idx[bg]] += 1.0
    return v


vocab_idx = build_vocab(dsets["train"]["sequence"])
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}


def encode(split):
    X = np.stack([vectorise(s, vocab_idx) for s in dsets[split]["sequence"]])
    y = np.array([label2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y


X_train, y_train = encode("train")
X_dev, y_dev = encode("dev")
X_test, y_test = encode("test")


class NgramDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.X[idx]), "y": torch.tensor(self.y[idx])}


batch_size = 128
train_loader = DataLoader(
    NgramDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(NgramDataset(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(NgramDataset(X_test, y_test), batch_size=batch_size)


# ---------------- Soft Decision Tree ------------------------------------------
class SoftDecisionTree(nn.Module):
    """depth d soft decision tree for classification"""

    def __init__(self, in_dim, num_classes, depth=3):
        super().__init__()
        self.depth = depth
        n_internal = 2**depth - 1
        n_leaves = 2**depth
        self.gate_w = nn.Parameter(torch.randn(n_internal, in_dim) * 0.01)
        self.gate_b = nn.Parameter(torch.zeros(n_internal))
        self.leaf_logits = nn.Parameter(torch.zeros(n_leaves, num_classes))

    def forward(self, x):
        batch = x.size(0)
        # prob for each node
        node_prob = x.new_ones(batch, 1)  # root prob
        leaf_probs = []
        idx_internal = 0
        for d in range(self.depth):
            n_nodes = 2**d
            w = self.gate_w[idx_internal : idx_internal + n_nodes]
            b = self.gate_b[idx_internal : idx_internal + n_nodes]
            g = torch.sigmoid(x @ w.t() + b)  # [batch,n_nodes]
            left = node_prob * g
            right = node_prob * (1 - g)
            node_prob = torch.cat([left, right], dim=1)  # probs for next depth
            idx_internal += n_nodes
        leaf_probs = node_prob  # [batch, n_leaves]
        logits = leaf_probs @ self.leaf_logits  # mix leaf logits
        return logits, leaf_probs

    # hard path prediction for rule extraction
    def hard_predict(self, x):
        with torch.no_grad():
            batch = x.size(0)
            path_idx = torch.zeros(batch, dtype=torch.long, device=x.device)
            idx_internal = 0
            node_prob = torch.ones(batch, 1, device=x.device)
            for d in range(self.depth):
                n_nodes = 2**d
                w = self.gate_w[idx_internal : idx_internal + n_nodes]
                b = self.gate_b[idx_internal : idx_internal + n_nodes]
                g = torch.sigmoid(x @ w.t() + b)  # [batch,n_nodes]
                # which node am I in?
                node_indices = (
                    path_idx >> (self.depth - 1 - d)
                ) & 1  # 0 left/right? Not needed, simpler:
                # compute hard choice for each sample
                choices = (g > 0.5).long()  # left=1, but we need left=1? We'll map:
                # we need mapping per sample to node index
                new_path = []
                for i in range(batch):
                    current = path_idx[i]
                    node = (current << 1) | choices[
                        i, 0
                    ]  # approximate; easier: we just manually walk
                # easier: We'll compute probabilities to leaves then argmax
            # simplified: use leaf_probs argmax
        logits, _ = self.forward(x)
        return torch.argmax(logits, 1)  # fallback

    # To simplify, rule fidelity will compare leaf argmax vs leaf mixed probability argmax
    def hard_leaf_pred(self, x):
        with torch.no_grad():
            _, leaf_probs = self.forward(x)
            leaf_idx = torch.argmax(leaf_probs, 1)  # most likely leaf
            leaf_logits = self.leaf_logits[leaf_idx]
            return torch.argmax(leaf_logits, 1)


# ------------------- training utilities ---------------------------------------
criterion = nn.CrossEntropyLoss()
l1_lambda = 1e-4


def evaluate(model, loader):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    all_soft_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits, _ = model(batch["x"])
            loss = criterion(logits, batch["y"])
            preds = logits.argmax(1)
            total += batch["y"].size(0)
            correct += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            all_soft_preds.append(preds.cpu())
    return correct / total, loss_sum / total, torch.cat(all_soft_preds)


# ------------------- experiment dict ------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fid": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
    }
}

# ------------------- training loop --------------------------------------------
epochs = 20
model = SoftDecisionTree(len(vocab_idx), len(labels), depth=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    model.train()
    seen = correct = 0
    running_loss = 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits, _ = model(batch["x"])
        ce_loss = criterion(logits, batch["y"])
        l1_loss = sum(torch.sum(torch.abs(p)) for p in model.gate_w)
        loss = ce_loss + l1_lambda * l1_loss
        loss.backward()
        optimizer.step()
        running_loss += ce_loss.item() * batch["y"].size(0)
        correct += (logits.argmax(1) == batch["y"]).sum().item()
        seen += batch["y"].size(0)
    train_loss = running_loss / seen
    train_acc = correct / seen
    val_acc, val_loss, _ = evaluate(model, dev_loader)
    # rule fidelity: compare hard leaf pred vs soft pred
    model.eval()
    all_soft = []
    all_hard = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            soft = model(batch["x"])[0].argmax(1)
            hard = model.hard_leaf_pred(batch["x"])
            all_soft.append(soft.cpu())
            all_hard.append(hard.cpu())
    rfa = (torch.cat(all_soft) == torch.cat(all_hard)).float().mean().item()
    # store metrics
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["rule_fid"].append(rfa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    print(f"  train_acc={train_acc:.3f} val_acc={val_acc:.3f} RFA={rfa:.3f}")

# ------------------- test evaluation ------------------------------------------
test_acc, test_loss, _ = evaluate(model, test_loader)
print(f"\nTest accuracy: {test_acc:.3f}")

# predictions for storage
model.eval()
preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds.append(model(batch["x"])[0].argmax(1).cpu())
experiment_data["SPR_BENCH"]["predictions"] = torch.cat(preds).numpy()

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
