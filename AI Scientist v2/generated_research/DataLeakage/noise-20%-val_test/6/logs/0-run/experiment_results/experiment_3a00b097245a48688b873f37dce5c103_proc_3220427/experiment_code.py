import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datasets import DatasetDict, load_dataset

# ------------------- REPRODUCIBILITY -------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ------------------- WORK DIR --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- DATA ---------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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


# --------------- n-gram vectoriser ---------------------
def build_vocab(seqs: List[str]):
    unis, bis = set(), set()
    for s in seqs:
        unis.update(s)
        bis.update([s[i : i + 2] for i in range(len(s) - 1)])
    vocab = sorted(list(unis)) + sorted(list(bis))
    return {tok: i for i, tok in enumerate(vocab)}


def vectorise(seq: str, idx: Dict[str, int]) -> np.ndarray:
    v = np.zeros(len(idx), dtype=np.float32)
    for c in seq:
        if c in idx:
            v[idx[c]] += 1.0
    for i in range(len(seq) - 1):
        bg = seq[i : i + 2]
        if bg in idx:
            v[idx[bg]] += 1.0
    return v


train_seqs = dsets["train"]["sequence"]
vocab_idx = build_vocab(train_seqs)
num_feats = len(vocab_idx)
print(f"Feature size: {num_feats}")

labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Classes: {labels}")


def encode_split(split):
    X = np.stack([vectorise(s, vocab_idx) for s in dsets[split]["sequence"]]).astype(
        np.float32
    )
    y = np.array([label2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y


X_train_raw, y_train = encode_split("train")
X_dev_raw, y_dev = encode_split("dev")
X_test_raw, y_test = encode_split("test")

# --------------- Z-SCORE SCALING -----------------------
mean = X_train_raw.mean(axis=0, keepdims=True)
std = X_train_raw.std(axis=0, keepdims=True)
std[std < 1e-6] = 1.0  # avoid div-by-zero


def zscore(x):
    return (x - mean) / std


X_train_z = zscore(X_train_raw)
X_dev_z = zscore(X_dev_raw)
X_test_z = zscore(X_test_raw)

# dictionary of feature variants
feature_sets = {
    "raw": (X_train_raw, X_dev_raw, X_test_raw),
    "zscore": (X_train_z, X_dev_z, X_test_z),
}


# ---------------- DATASET WRAPPER ----------------------
class NgramDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.X[idx]), "y": torch.tensor(self.y[idx])}


batch_size = 128


# -------------------- MODEL ----------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()

# ------------- EXPERIMENT DATA STRUCTURE --------------
experiment_data = {
    "feature_scale": {
        "SPR_BENCH": {
            "configs": [],
            "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_test.tolist(),
            "best_config": "",
        }
    }
}


# ------------- HELPER: EVALUATION ----------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    all_logits = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        preds = logits.argmax(1)
        correct += (preds == batch["y"]).sum().item()
        tot += batch["y"].size(0)
        loss_sum += loss.item() * batch["y"].size(0)
        all_logits.append(logits.cpu())
    return correct / tot, loss_sum / tot, torch.cat(all_logits)


# ------------- TRAINING -------------------------------
grid = [("adam", None)] + [("sgd", m) for m in (0.0, 0.5, 0.9)]
lr_map = {"adam": 1e-3, "sgd": 0.1}
epochs, top_k = 10, 10

best_val_acc, best_pred, best_cfg_name = -1.0, None, ""

for feat_name, (X_tr, X_dv, X_te) in feature_sets.items():
    # build loaders for this variant
    train_loader = DataLoader(
        NgramDataset(X_tr, y_train), batch_size=batch_size, shuffle=True
    )
    dev_loader = DataLoader(NgramDataset(X_dv, y_dev), batch_size=batch_size)
    test_loader = DataLoader(NgramDataset(X_te, y_test), batch_size=batch_size)

    for opt_name, momentum in grid:
        base_cfg = opt_name if opt_name == "adam" else f"sgd_m{momentum}"
        cfg_name = f"{feat_name}_{base_cfg}"
        print(f"\n===== Training {cfg_name} =====")
        experiment_data["feature_scale"]["SPR_BENCH"]["configs"].append(cfg_name)

        model = LogReg(num_feats, num_classes).to(device)
        optimizer = (
            optim.Adam(model.parameters(), lr=lr_map["adam"])
            if opt_name == "adam"
            else optim.SGD(model.parameters(), lr=lr_map["sgd"], momentum=momentum)
        )

        run_train_acc, run_val_acc, run_rule_fid = [], [], []
        run_train_loss, run_val_loss = [], []

        for ep in range(1, epochs + 1):
            model.train()
            seen, correct, loss_sum = 0, 0, 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = model(batch["x"])
                loss = criterion(logits, batch["y"])
                loss.backward()
                optimizer.step()
                preds = logits.argmax(1)
                correct += (preds == batch["y"]).sum().item()
                seen += batch["y"].size(0)
                loss_sum += loss.item() * batch["y"].size(0)
            train_loss = loss_sum / seen
            train_acc = correct / seen
            val_acc, val_loss, val_logits = evaluate(model, dev_loader)

            # rule fidelity
            W = model.linear.weight.detach().cpu().numpy()
            b = model.linear.bias.detach().cpu().numpy()
            W_trunc = np.zeros_like(W)
            for c in range(num_classes):
                idxs = np.argsort(-np.abs(W[c]))[:top_k]
                W_trunc[c, idxs] = W[c, idxs]
            lin_full = torch.from_numpy((X_dv @ W.T) + b)
            lin_trunc = torch.from_numpy((X_dv @ W_trunc.T) + b)
            rule_fid = (lin_trunc.argmax(1) == lin_full.argmax(1)).float().mean().item()

            run_train_acc.append(train_acc)
            run_val_acc.append(val_acc)
            run_rule_fid.append(rule_fid)
            run_train_loss.append(train_loss)
            run_val_loss.append(val_loss)

            print(
                f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
            )

        # store run
        ed = experiment_data["feature_scale"]["SPR_BENCH"]
        ed["metrics"]["train_acc"].append(run_train_acc)
        ed["metrics"]["val_acc"].append(run_val_acc)
        ed["metrics"]["rule_fidelity"].append(run_rule_fid)
        ed["losses"]["train"].append(run_train_loss)
        ed["losses"]["val"].append(run_val_loss)

        if run_val_acc[-1] > best_val_acc:
            best_val_acc = run_val_acc[-1]
            best_cfg_name = cfg_name
            test_acc, test_loss, test_logits = evaluate(model, test_loader)
            best_pred = test_logits.argmax(1).cpu().numpy()
            print(f"*** New best config: {cfg_name} test_acc={test_acc:.3f}")

# --------------- SAVE RESULTS --------------------------
ed = experiment_data["feature_scale"]["SPR_BENCH"]
ed["predictions"] = best_pred.tolist()
ed["best_config"] = best_cfg_name


def to_np(o):
    return np.array(o, dtype=object)


for k in ["train_acc", "val_acc", "rule_fidelity"]:
    ed["metrics"][k] = to_np(ed["metrics"][k])
for k in ["train", "val"]:
    ed["losses"][k] = to_np(ed["losses"][k])
ed["predictions"] = np.array(ed["predictions"])
ed["ground_truth"] = np.array(ed["ground_truth"])

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nBest configuration: {best_cfg_name} with dev_acc={best_val_acc:.3f}")
print("Saved all results to", os.path.join(working_dir, "experiment_data.npy"))
