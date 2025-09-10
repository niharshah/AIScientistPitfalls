import os, pathlib, numpy as np, torch, math, collections, time
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- load SPR_BENCH -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict()
    for split in ["train", "dev", "test"]:
        ds[split] = _l(f"{split}.csv")
    return ds


possible_paths = [
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
    pathlib.Path("SPR_BENCH/"),
]
for p in possible_paths:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ----------------- build trigram vocab -----------------
NGRAM_N = 3
TOP_K = 4000  # limit feature size for speed / memory

freq = collections.Counter()
for seq in spr["train"]["sequence"]:
    for i in range(len(seq) - NGRAM_N + 1):
        freq[seq[i : i + NGRAM_N]] += 1
most_common = [ng for ng, _ in freq.most_common(TOP_K)]
ngram2idx = {ng: i for i, ng in enumerate(most_common)}
vocab_dim = len(ngram2idx)
num_classes = len(set(spr["train"]["label"]))
print(f"Trigram features: {vocab_dim}  |  Classes: {num_classes}")


def encode(seq):
    vec = torch.zeros(vocab_dim, dtype=torch.float32)
    for i in range(len(seq) - NGRAM_N + 1):
        ng = seq[i : i + NGRAM_N]
        if ng in ngram2idx:
            vec[ngram2idx[ng]] += 1.0
    return vec


# ----------------- torch Dataset -----------------
class SPRNgramDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs.to(device), "y": ys.to(device)}


batch_size = 256
train_dl = DataLoader(
    SPRNgramDataset(spr["train"]), batch_size, shuffle=True, collate_fn=collate_fn
)
dev_dl = DataLoader(
    SPRNgramDataset(spr["dev"]), batch_size, shuffle=False, collate_fn=collate_fn
)
test_dl = DataLoader(
    SPRNgramDataset(spr["test"]), batch_size, shuffle=False, collate_fn=collate_fn
)

# ----------------- model -----------------
model = nn.Sequential(nn.Linear(vocab_dim, num_classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# ----------------- containers for logging -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "REA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


def run_epoch(model, loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["y"].size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(batch["y"].cpu().tolist())
    acc = accuracy_score(gts, preds)
    return tot_loss / len(loader.dataset), acc, preds, gts


EPOCHS = 10
best_val_acc, best_state = 0.0, None
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, _, _ = run_epoch(model, train_dl, train=True)
    val_loss, val_acc, _, _ = run_epoch(model, dev_dl, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc = {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

# ----------------- load best -----------------
model.load_state_dict(best_state)

# ----------------- RULE EXTRACTION -----------------
with torch.no_grad():
    W = model[0].weight.cpu().numpy()  # shape [num_classes, vocab_dim]
top_rules = {}
TOP_RULES_PER_LABEL = 20
for lbl in range(num_classes):
    # score = weight for lbl minus average other class weight → discriminative
    diff = W[lbl] - W.mean(axis=0)
    idxs = diff.argsort()[-TOP_RULES_PER_LABEL:][::-1]
    top_rules[lbl] = [(most_common[i], float(diff[i])) for i in idxs]


# ----------------- Rule-based inference for REA -----------------
def rule_predict(seq):
    scores = np.zeros(num_classes)
    seq_ngrams = set(seq[i : i + NGRAM_N] for i in range(len(seq) - NGRAM_N + 1))
    for lbl in range(num_classes):
        for ng, w in top_rules[lbl]:
            if ng in seq_ngrams:
                scores[lbl] += w
    return int(scores.argmax())


test_sequences = spr["test"]["sequence"]
rule_preds = [rule_predict(s) for s in test_sequences]
REA = accuracy_score(spr["test"]["label"], rule_preds)
print(f"Rule Extraction Accuracy (REA): {REA:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["REA"] = REA

# ----------------- final test accuracy -----------------
test_loss, test_acc, preds, gts = run_epoch(model, test_dl, train=False)
print(f"Test accuracy (full model): {test_acc:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# ----------------- save experiment data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
