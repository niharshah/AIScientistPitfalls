import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import DatasetDict
import pathlib
from typing import List

# ---------------- GPU / device handling ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Data loading helpers -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset, DatasetDict

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def complexity_weight(sequence: str) -> int:
    return count_color_variety(sequence) * count_shape_variety(sequence)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------------- Simple dataset class -----------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, vocab, label2id):
        self.seqs = hf_split["sequence"]
        self.labels = [label2id[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = self.seqs[idx].split()
        ids = [self.vocab[t] for t in tokens]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def build_vocab(seqs: List[str]):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def collate(batch):
    maxlen = max(len(x["ids"]) for x in batch)
    ids = []
    labels = []
    raws = []
    for b in batch:
        pad_len = maxlen - len(b["ids"])
        padded = torch.cat([b["ids"], torch.zeros(pad_len, dtype=torch.long)])
        ids.append(padded)
        labels.append(b["label"])
        raws.append(b["raw_seq"])
    return {"ids": torch.stack(ids), "label": torch.stack(labels), "raw_seq": raws}


# ---------------- Model -----------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lin = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        emb = self.emb(x)  # [B,L,D]
        mask = (x != 0).unsqueeze(-1)  # [B,L,1]
        summed = (emb * mask).sum(1)  # [B,D]
        lengths = mask.sum(1).clamp(min=1)  # avoid div 0
        avg = summed / lengths
        return self.lin(avg)


# ---------------- Training utilities -----------------
def run_epoch(model, loader, optim=None, criterion=None):
    train = optim is not None
    total_loss, n = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["ids"])
        loss = criterion(logits, batch["label"]) if criterion else None
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item() * batch["label"].size(0) if loss else 0
        n += batch["label"].size(0)
        preds = logits.argmax(-1).cpu().tolist()
        labels = batch["label"].cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_seqs.extend(batch["raw_seq"])
    avg_loss = total_loss / n if n else 0.0
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cpx = complexity_weighted_accuracy(all_seqs, all_labels, all_preds)
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    return avg_loss, acc, cwa, swa, cpx


# ---------------- Main experiment -----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

vocab = build_vocab(spr["train"]["sequence"])
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
print(f"Vocab size {len(vocab)}, num classes {len(labels)}")

train_set = SPRDataset(spr["train"], vocab, label2id)
dev_set = SPRDataset(spr["dev"], vocab, label2id)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_set, batch_size=512, shuffle=False, collate_fn=collate)

model = AvgEmbedClassifier(len(vocab), emb_dim=64, n_classes=len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_cwa, tr_swa, tr_cpx = run_epoch(
        model, train_loader, optim=optimizer, criterion=criterion
    )
    val_loss, val_acc, val_cwa, val_swa, val_cpx = run_epoch(
        model, dev_loader, optim=None, criterion=criterion
    )
    print(
        f"Epoch {epoch}: "
        f"validation_loss = {val_loss:.4f}, "
        f"val_acc = {val_acc:.4f}, "
        f"CWA = {val_cwa:.4f}, SWA = {val_swa:.4f}, CpxWA = {val_cpx:.4f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        (epoch, tr_acc, tr_cwa, tr_swa, tr_cpx)
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (epoch, val_acc, val_cwa, val_swa, val_cpx)
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
