# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
import pathlib
from typing import Dict, List
from datasets import DatasetDict

# ------------------------------------------------------------------
# 0. GPU / CPU handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ------------------------------------------------------------------
# 1. Data loading ---------------------------------------------------
# Assumes SPR_BENCH exists at this location. Change if necessary.
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr)


# ------------------------------------------------------------------
# 2. Build vocabulary & encode -------------------------------------
def build_vocab(dsets) -> Dict[str, int]:
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(list(s))
    vocab = {ch: i + 1 for i, ch in enumerate(sorted(chars))}  # 0 reserved for PAD
    return vocab


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1  # +PAD
print(f"Vocab size: {vocab_size}")


def encode_sequence(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[ch] for ch in seq]


# compute max length
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())
print(f"Max sequence length: {max_len}")


def pad(seq_ids: List[int], max_len: int) -> List[int]:
    if len(seq_ids) >= max_len:
        return seq_ids[:max_len]
    return seq_ids + [0] * (max_len - len(seq_ids))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, max_len):
        self.labels = hf_split["label"]
        self.seqs = hf_split["sequence"]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq_ids = pad(encode_sequence(self.seqs[idx], self.vocab), self.max_len)
        return {
            "input_ids": torch.tensor(seq_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


train_ds = SPRTorchDataset(spr["train"], vocab, max_len)
dev_ds = SPRTorchDataset(spr["dev"], vocab, max_len)
test_ds = SPRTorchDataset(spr["test"], vocab, max_len)

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# ------------------------------------------------------------------
# 3. Model ----------------------------------------------------------
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)  # h: [2, B, H]
        h = torch.cat([h[0], h[1]], dim=1)  # [B, 2H]
        out = self.fc(h).squeeze(1)  # [B]
        return out


model = GRUBaseline(vocab_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# ------------------------------------------------------------------
# 4. Experiment data container -------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------
# 5. Training loop --------------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    # ---- train ---
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
    train_loss = running_loss / len(train_ds)
    # Approx train MCC on one pass (optional quick)
    with torch.no_grad():
        preds_all, labels_all = [], []
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"])
            preds_all.append((logits.sigmoid() > 0.5).cpu().numpy())
            labels_all.append(batch["labels"].cpu().numpy())
        train_mcc = matthews_corrcoef(
            np.concatenate(labels_all), np.concatenate(preds_all)
        )
    # ---- validation ---
    model.eval()
    val_loss = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            preds_all.append((logits.sigmoid() > 0.5).cpu().numpy())
            labels_all.append(batch["labels"].cpu().numpy())
    val_loss /= len(dev_ds)
    val_mcc = matthews_corrcoef(np.concatenate(labels_all), np.concatenate(preds_all))

    # store metrics
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_mcc)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_mcc)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_MCC={val_mcc:.4f}"
    )

# ------------------------------------------------------------------
# 6. Final evaluation on test --------------------------------------
model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = model(batch["input_ids"])
        preds_all.append((logits.sigmoid() > 0.5).cpu().numpy())
        labels_all.append(batch["labels"].cpu().numpy())
preds_all = np.concatenate(preds_all)
labels_all = np.concatenate(labels_all)
test_mcc = matthews_corrcoef(labels_all, preds_all)
print(f"Test MCC: {test_mcc:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds_all
experiment_data["SPR_BENCH"]["ground_truth"] = labels_all

# ------------------------------------------------------------------
# 7. Save experiment data ------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
