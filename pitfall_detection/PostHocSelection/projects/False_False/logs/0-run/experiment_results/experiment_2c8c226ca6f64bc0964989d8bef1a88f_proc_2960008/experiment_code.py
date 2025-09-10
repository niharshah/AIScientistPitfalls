# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter
from typing import List, Dict
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macroF1": [], "val_macroF1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# ----------- DATA --------------------------------------------------
def resolve_spr_path() -> pathlib.Path:
    """Return Path to SPR_BENCH that actually exists with train.csv."""
    candidates = []
    if "SPR_BENCH_PATH" in os.environ:
        candidates.append(os.environ["SPR_BENCH_PATH"])
    cwd = pathlib.Path.cwd()
    candidates += [
        cwd / "SPR_BENCH",
        cwd.parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for cand in candidates:
        p = pathlib.Path(cand)
        if (p / "train.csv").exists():
            print(f"Found SPR_BENCH dataset at {p.resolve()}")
            return p.resolve()
    raise FileNotFoundError(
        "SPR_BENCH dataset not found. Set env SPR_BENCH_PATH or place csvs in ./SPR_BENCH"
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",  # treat csv as a single split
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


spr_root = resolve_spr_path()
spr = load_spr_bench(spr_root)
print("Loaded SPR_BENCH with sizes:", {k: len(v) for k, v in spr.items()})


# ----------- VOCAB -------------------------------------------------
def tokenize(seq: str) -> List[str]:
    return seq.strip().split()


all_tokens = [tok for seq in spr["train"]["sequence"] for tok in tokenize(seq)]
vocab_counter = Counter(all_tokens)
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = stoi["<PAD>"], stoi["<UNK>"]

all_labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(all_labels)}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, unk_idx) for tok in tokenize(seq)]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(x["input_ids"]) for x in batch]
    maxlen = max(lengths)
    input_ids = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "label": labels}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ----------- MODEL -------------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_labels, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(emb_dim, num_labels)
        self.pad = pad_idx

    def forward(self, x):
        mask = (x != self.pad).unsqueeze(-1)
        emb = self.emb(x)
        mean = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(mean))


model = MeanPoolClassifier(len(vocab), 64, len(all_labels), pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------- TRAIN LOOP -------------------------------------------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    # ---- train ----
    model.train()
    train_loss, train_preds, train_trues = 0.0, [], []
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch["label"].size(0)
        train_preds.extend(logits.argmax(1).cpu().numpy())
        train_trues.extend(batch["label"].cpu().numpy())
    train_loss /= len(train_loader.dataset)
    train_macro = f1_score(train_trues, train_preds, average="macro")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_macroF1"].append(train_macro)

    # ---- validation ----
    model.eval()
    val_loss, val_preds, val_trues = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["label"].size(0)
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_trues.extend(batch["label"].cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_macro = f1_score(val_trues, val_preds, average="macro")
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_macroF1"].append(val_macro)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, Val MacroF1 = {val_macro:.4f}"
    )

# -------- store predictions / GT for dev split --------------------
experiment_data["SPR_BENCH"]["predictions"] = val_preds
experiment_data["SPR_BENCH"]["ground_truth"] = val_trues


# --------- Optional SWA / CWA -------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs: List[str], y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0


def color_weighted_accuracy(seqs: List[str], y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0


swa = shape_weighted_accuracy(spr["dev"]["sequence"], val_trues, val_preds)
cwa = color_weighted_accuracy(spr["dev"]["sequence"], val_trues, val_preds)
print(f"Dev SWA: {swa:.4f} | CWA: {cwa:.4f}")

# --------- SAVE everything ----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
