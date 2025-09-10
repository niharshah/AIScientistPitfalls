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


# --------------- n-gram vectoriser ---------------------
def build_vocab(seqs: List[str]):
    unis, bis = set(), set()
    for s in seqs:
        unis.update(s)
        bis.update([s[i : i + 2] for i in range(len(s) - 1)])
    vocab = sorted(unis) + sorted(bis)
    return {tok: i for i, tok in enumerate(vocab)}


def vectorise(seq: str, idx: Dict[str, int]):
    v = np.zeros(len(idx), dtype=np.float32)
    for c in seq:
        if c in idx:
            v[idx[c]] += 1.0
    for i in range(len(seq) - 1):
        bg = seq[i : i + 2]
        if bg in idx:
            v[idx[bg]] += 1.0
    return v


vocab_idx = build_vocab(dsets["train"]["sequence"])
num_feats = len(vocab_idx)
labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)


def encode_split(split):
    X = np.stack([vectorise(s, vocab_idx) for s in dsets[split]["sequence"]])
    y = np.array([label2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y


X_train, y_train = encode_split("train")
X_dev, y_dev = encode_split("dev")
X_test, y_test = encode_split("test")


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


# -------------------- MODEL ----------------------------
class LogReg(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


criterion = nn.CrossEntropyLoss()

# ------------ EXPERIMENT DATA STRUCTURE ---------------
experiment_data = {
    "feature_dropout": {
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
def evaluate(model, loader):
    model.eval()
    tot = 0
    corr = 0
    loss_sum = 0.0
    logits_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            _, preds = torch.max(logits, 1)
            tot += batch["y"].size(0)
            corr += (preds == batch["y"]).sum().item()
            loss_sum += loss.item() * batch["y"].size(0)
            logits_all.append(logits.cpu())
    return corr / tot, loss_sum / tot, torch.cat(logits_all)


# ------------- GRID SETUP ------------------------------
optimizer_grid = [("adam", None)] + [("sgd", m) for m in (0.0, 0.5, 0.9)]
lr_map = {"adam": 1e-3, "sgd": 0.1}
dropout_ps = [0.0, 0.2]  # 0.0 = baseline, 0.2 = feature dropout ablation
epochs, top_k = 10, 10

best_val_acc = -1.0
best_pred = None
best_cfg = ""

for p in dropout_ps:
    for opt_name, momentum in optimizer_grid:
        cfg_name = f"drop{p}_{opt_name if opt_name=='adam' else f'sgd_m{momentum}'}"
        print(f"\n===== Training config: {cfg_name} =====")
        ed = experiment_data["feature_dropout"]["SPR_BENCH"]
        ed["configs"].append(cfg_name)

        model = LogReg(num_feats, num_classes).to(device)
        optimizer = (
            optim.Adam(model.parameters(), lr=lr_map["adam"])
            if opt_name == "adam"
            else optim.SGD(model.parameters(), lr=lr_map["sgd"], momentum=momentum)
        )

        run_train_acc, run_val_acc, run_rule_fid = [], [], []
        run_train_loss, run_val_loss = [], []

        for epoch in range(1, epochs + 1):
            model.train()
            seen = 0
            correct = 0
            run_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # -------- Feature-level dropout ---------
                x_in = batch["x"]
                if p > 0.0:
                    mask = (torch.rand_like(x_in) > p).float()
                    x_in = x_in * mask
                optimizer.zero_grad()
                logits = model(x_in)
                loss = criterion(logits, batch["y"])
                loss.backward()
                optimizer.step()
                run_loss += loss.item() * batch["y"].size(0)
                correct += (torch.argmax(logits, 1) == batch["y"]).sum().item()
                seen += batch["y"].size(0)

            train_loss = run_loss / seen
            train_acc = correct / seen
            val_acc, val_loss, _ = evaluate(model, dev_loader)

            # ----- rule fidelity -----
            W = model.linear.weight.detach().cpu().numpy()
            b = model.linear.bias.detach().cpu().numpy()
            W_trunc = np.zeros_like(W)
            for c in range(num_classes):
                idxs = np.argsort(-np.abs(W[c]))[:top_k]
                W_trunc[c, idxs] = W[c, idxs]
            lin_full = torch.from_numpy((X_dev @ W.T) + b)
            lin_trunc = torch.from_numpy((X_dev @ W_trunc.T) + b)
            rule_fid = (
                (torch.argmax(lin_trunc, 1) == torch.argmax(lin_full, 1))
                .float()
                .mean()
                .item()
            )

            run_train_acc.append(train_acc)
            run_val_acc.append(val_acc)
            run_rule_fid.append(rule_fid)
            run_train_loss.append(train_loss)
            run_val_loss.append(val_loss)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} rule_fid={rule_fid:.3f}"
            )

        ed["metrics"]["train_acc"].append(run_train_acc)
        ed["metrics"]["val_acc"].append(run_val_acc)
        ed["metrics"]["rule_fidelity"].append(run_rule_fid)
        ed["losses"]["train"].append(run_train_loss)
        ed["losses"]["val"].append(run_val_loss)

        if run_val_acc[-1] > best_val_acc:
            best_val_acc = run_val_acc[-1]
            best_cfg = cfg_name
            test_acc, test_loss, test_logits = evaluate(model, test_loader)
            best_pred = torch.argmax(test_logits, 1).cpu().numpy()
            print(f"*** New best cfg: {cfg_name} with test_acc={test_acc:.3f}")

# ------------ FINAL SAVE --------------------------------
ed = experiment_data["feature_dropout"]["SPR_BENCH"]
ed["predictions"] = best_pred.tolist()
ed["best_config"] = best_cfg


# convert to numpy arrays
def to_np(o):
    return np.array(o, dtype=object)


for k in ["train_acc", "val_acc", "rule_fidelity"]:
    ed["metrics"][k] = to_np(ed["metrics"][k])
for k in ["train", "val"]:
    ed["losses"][k] = to_np(ed["losses"][k])
ed["predictions"] = np.array(ed["predictions"])
ed["ground_truth"] = np.array(ed["ground_truth"])

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nBest overall configuration: {best_cfg} with dev_acc={best_val_acc:.3f}")
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
