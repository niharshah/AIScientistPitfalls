# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, random, time, math
import numpy as np
from typing import List, Dict
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- dataset helpers ----------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return float(sum(correct)) / float(sum(weights)) if sum(weights) else 0.0


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return float(sum(correct)) / float(sum(weights)) if sum(weights) else 0.0


def dwhs(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ---------- load data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", {k: len(v) for k, v in spr.items()})


# ---------- glyph clustering ----------
def token_feature(tok: str) -> List[float]:
    chars = [ord(c) for c in tok]
    first = chars[0]
    rest_mean = sum(chars[1:]) / len(chars[1:]) if len(chars) > 1 else 0.0
    return [first, rest_mean]


all_tokens = set()
for seq in spr["train"]["sequence"]:
    all_tokens.update(seq.strip().split())
all_tokens = sorted(list(all_tokens))
X = np.array([token_feature(t) for t in all_tokens])
k = max(8, int(math.sqrt(len(all_tokens))))  # simple heuristic
print(f"Clustering {len(all_tokens)} unique glyphs into {k} clusters …")
km = KMeans(n_clusters=k, random_state=0, n_init="auto")
clusters = km.fit_predict(X)
glyph2cluster = {tok: int(c) for tok, c in zip(all_tokens, clusters)}
print("Done clustering.")


# ---------- dataset class ----------
class SPRClusteredDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.y2id = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tok_ids = [
            glyph2cluster.get(t, 0) + 1  # +1 to reserve 0 for PAD
            for t in self.seqs[idx].strip().split()
        ]
        return {
            "input": torch.tensor(tok_ids, dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    lengths = [len(x["input"]) for x in batch]
    maxlen = max(lengths)
    inputs = []
    for x in batch:
        pad_len = maxlen - len(x["input"])
        inputs.append(torch.cat([x["input"], torch.zeros(pad_len, dtype=torch.long)]))
    inputs = torch.stack(inputs)
    labels = torch.stack([x["label"] for x in batch])
    raw_seqs = [x["raw_seq"] for x in batch]
    return {
        "input": inputs.to(device),
        "label": labels.to(device),
        "len": torch.tensor(lengths, dtype=torch.long).to(device),
        "raw_seq": raw_seqs,
    }


train_ds = SPRClusteredDataset(spr["train"])
dev_ds = SPRClusteredDataset(spr["dev"])
test_ds = SPRClusteredDataset(spr["test"])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

num_labels = len(set(spr["train"]["label"]))
vocab_size = k + 1 + 1  # clusters + pad + OOV
print(f"num_labels={num_labels}, vocab_size={vocab_size}")


# ---------- model ----------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        logits = self.fc(h.squeeze(0))
        return logits


model = GRUClassifier(vocab_size, 32, 64, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# ---------- training ----------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input"], batch["len"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["label"].size(0)
    train_loss = epoch_loss / len(train_ds)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))

    # ---- validation ----
    model.eval()
    val_loss = 0.0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            logits = model(batch["input"], batch["len"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["label"].size(0)
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            labels = batch["label"].cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(batch["raw_seq"])
    val_loss /= len(dev_ds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    val_dwhs = dwhs(cwa, swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, cwa, swa, val_dwhs))

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA={cwa:.3f} SWA={swa:.3f} DWHS={val_dwhs:.3f}"
    )

# ---------- test evaluation ----------
model.eval()
test_preds, test_labels, test_seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input"], batch["len"])
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        labels = batch["label"].cpu().numpy().tolist()
        test_preds.extend(preds)
        test_labels.extend(labels)
        test_seqs.extend(batch["raw_seq"])
cwa = color_weighted_accuracy(test_seqs, test_labels, test_preds)
swa = shape_weighted_accuracy(test_seqs, test_labels, test_preds)
test_dwhs = dwhs(cwa, swa)
print(f"TEST  | CWA={cwa:.3f} SWA={swa:.3f} DWHS={test_dwhs:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
