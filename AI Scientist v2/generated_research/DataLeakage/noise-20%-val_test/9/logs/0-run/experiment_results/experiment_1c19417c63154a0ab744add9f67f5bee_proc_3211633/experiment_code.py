import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time
import pathlib
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------ Device ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Hyper-parameters ------------------ #
MAX_UNI, MAX_BI, MAX_TRI = 100, 1000, 2000  # size of kept n-gram sets
BATCH_SIZE, VAL_BATCH = 256, 512
LR, EPOCHS = 5e-3, 12
L1_LAMBDA = 1e-4  # sparsity weight
TOPK_RULE = 3  # n-grams per class for rule
rng = np.random.default_rng(42)


# ------------------ Dataset loading ------------------ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _ld(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------------ Build n-gram vocabulary ------------------ #
def get_ngrams(seq: str, n: int) -> List[str]:
    return [seq[i : i + n] for i in range(len(seq) - n + 1)]


uni_ctr, bi_ctr, tri_ctr = Counter(), Counter(), Counter()
for s in spr["train"]["sequence"]:
    uni_ctr.update(get_ngrams(s, 1))
    bi_ctr.update(get_ngrams(s, 2))
    tri_ctr.update(get_ngrams(s, 3))

uni_vocab = [g for g, _ in uni_ctr.most_common(MAX_UNI)]
bi_vocab = [g for g, _ in bi_ctr.most_common(MAX_BI)]
tri_vocab = [g for g, _ in tri_ctr.most_common(MAX_TRI)]
ngram2idx = {g: i for i, g in enumerate(uni_vocab + bi_vocab + tri_vocab)}
idx2ngram = {i: g for g, i in ngram2idx.items()}
feat_dim = len(ngram2idx)
print(f"Feature dimension: {feat_dim}")


# ------------------ Vectorisation ------------------ #
def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(feat_dim, dtype=np.float32)
    L = len(seq)
    if L == 0:
        return vec
    for g in get_ngrams(seq, 1):
        if g in ngram2idx:
            vec[ngram2idx[g]] += 1.0
    for g in get_ngrams(seq, 2):
        if g in ngram2idx:
            vec[ngram2idx[g]] += 1.0
    for g in get_ngrams(seq, 3):
        if g in ngram2idx:
            vec[ngram2idx[g]] += 1.0
    vec /= L
    return vec


def prep_split(split):
    X = np.stack([seq_to_vec(s) for s in split["sequence"]])
    y = np.array(split["label"], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


X_tr, y_tr = prep_split(spr["train"])
X_dev, y_dev = prep_split(spr["dev"])
X_tst, y_tst = prep_split(spr["test"])
n_classes = int(max(y_tr.max(), y_dev.max(), y_tst.max()) + 1)
print(f"Classes: {n_classes}")

train_loader = DataLoader(
    TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=VAL_BATCH)
test_loader = DataLoader(TensorDataset(X_tst, y_tst), batch_size=VAL_BATCH)


# ------------------ Model ------------------ #
class NGramLinear(nn.Module):
    def __init__(self, in_dim: int, n_cls: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_cls)

    def forward(self, x):
        return self.linear(x)


model = NGramLinear(feat_dim, n_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ------------------ Metric store ------------------ #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "Rule_Fidelity": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------ Utils ------------------ #
def eval_split(loader):
    model.eval()
    tot, corr, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            tot += yb.size(0)
            corr += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
    return corr / tot, loss_sum / tot


def extract_rules(topk: int = TOPK_RULE) -> List[List[int]]:
    W = model.linear.weight.detach().cpu().numpy()  # (C,D)
    top_idx = np.argsort(W, axis=1)[:, -topk:]  # indices of largest weights
    return top_idx


def rule_predict(xvecs: torch.Tensor, rules: List[List[int]]) -> torch.Tensor:
    # xvecs : (B,D) float32
    counts = xvecs.cpu().numpy()
    preds = []
    for row in counts:
        scores = []
        for cls, idxs in enumerate(rules):
            scores.append(row[idxs].sum())
        preds.append(int(np.argmax(scores)))
    return torch.tensor(preds)


def rule_fidelity(loader):
    rules = extract_rules()
    total, match = 0, 0
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb_d = xb.to(device)
            logits = model(xb_d)
            full_preds = logits.argmax(1).cpu()
            rule_preds = rule_predict(xb, rules)
            match += (rule_preds == full_preds).sum().item()
            total += xb.size(0)
    return match / total


# ------------------ Training loop ------------------ #
for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss, ep_corr, ep_seen = 0.0, 0, 0
    t0 = time.time()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        outs = model(xb)
        ce_loss = criterion(outs, yb)
        l1_pen = sum(p.abs().sum() for p in model.parameters())
        loss = ce_loss + L1_LAMBDA * l1_pen
        loss.backward()
        opt.step()
        preds = outs.argmax(1)
        ep_loss += ce_loss.item() * yb.size(0)
        ep_corr += (preds == yb).sum().item()
        ep_seen += yb.size(0)
    train_acc = ep_corr / ep_seen
    train_loss = ep_loss / ep_seen
    val_acc, val_loss = eval_split(val_loader)
    fid = rule_fidelity(val_loader)
    # store
    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["Rule_Fidelity"].append(fid)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, RuleFid={fid:.3f}"
    )

# ------------------ Test evaluation ------------------ #
test_acc, test_loss = eval_split(test_loader)
test_fid = rule_fidelity(test_loader)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}, RuleFid={test_fid:.3f}")

# Predictions for persistence
model.eval()
all_p, all_g = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(1).cpu()
        all_p.append(preds)
        all_g.append(yb)
experiment_data["SPR_BENCH"]["predictions"] = torch.cat(all_p).numpy()
experiment_data["SPR_BENCH"]["ground_truth"] = torch.cat(all_g).numpy()

# ------------------ Save experiment data ------------------ #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
