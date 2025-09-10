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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-file script: GRU on SPR-BENCH with epoch-number tuning + early stopping.
"""
import os, pathlib, time, math, json, random, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# --------------------------- house-keeping ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------- experiment log scaffold -----------------
experiment_data = {
    "EPOCH_TUNING": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_rec = experiment_data["EPOCH_TUNING"]["SPR_BENCH"]  # shorthand


# --------------------------- metric helpers --------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def rule_signature(sequence: str) -> str:
    return " ".join(tok[0] for tok in sequence.strip().split() if tok)


# --------------------------- data loading ----------------------------
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


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# --------------------------- vocab & encoding ------------------------
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"


def build_vocab(dataset):
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    tokens = set(itertools.chain.from_iterable(seq.strip().split() for seq in dataset))
    for tok in sorted(tokens):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
print(f"Vocab size: {len(vocab)}")


def encode_sequence(seq: str, vocab=vocab):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.strip().split()]


label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print(f"Labels: {label_set}")


# --------------------------- torch datasets --------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "seq_enc": torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    seqs = [b["seq_enc"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=vocab[PAD_TOKEN]
    )
    return {"input_ids": padded, "labels": labels, "raw_seq": raw}


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr["train"]),
    SPRTorchDataset(spr["dev"]),
    SPRTorchDataset(spr["test"]),
)

train_loader = DataLoader(train_ds, 128, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, 256, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, 256, shuffle=False, collate_fn=collate_fn)


# --------------------------- model -----------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


model = GRUClassifier(len(vocab), 32, 64, len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Precompute signatures for NRGS
train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])


# --------------------------- evaluation ------------------------------
def evaluate(loader):
    model.eval()
    total, corr, loss_sum = 0, 0, 0.0
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            logits = model(inp)
            loss = criterion(logits, lab)
            loss_sum += loss.item() * len(lab)
            pred = logits.argmax(dim=-1)
            corr += (pred == lab).sum().item()
            total += len(lab)
            all_seq.extend(batch["raw_seq"])
            all_true.extend(lab.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
    acc = corr / total
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    novel_mask = [rule_signature(s) not in train_signatures for s in all_seq]
    novel_tot = sum(novel_mask)
    novel_corr = sum(
        int(p == t) for p, t, m in zip(all_pred, all_true, novel_mask) if m
    )
    nrgs = novel_corr / novel_tot if novel_tot else 0.0
    return loss_sum / total, acc, swa, cwa, nrgs, all_pred, all_true, all_seq


# --------------------------- training loop ---------------------------
MAX_EPOCHS, PATIENCE = 20, 3
best_val_loss, best_state, epochs_no_improve = math.inf, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
    train_loss = running_loss / len(train_ds)

    val_loss, val_acc, val_swa, val_cwa, val_nrgs, *_ = evaluate(dev_loader)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS}: "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"acc={val_acc:.3f}  SWA={val_swa:.3f}  CWA={val_cwa:.3f}  NRGS={val_nrgs:.3f}"
    )

    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train"].append({"epoch": epoch})
    exp_rec["metrics"]["val"].append(
        {
            "epoch": epoch,
            "acc": val_acc,
            "swa": val_swa,
            "cwa": val_cwa,
            "nrgs": val_nrgs,
        }
    )
    exp_rec["timestamps"].append(time.time())

    # --- early stopping bookkeeping ---
    if val_loss < best_val_loss - 1e-4:
        best_val_loss, best_state = val_loss, model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early.")
            break

# restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# --------------------------- final test eval -------------------------
test_loss, test_acc, test_swa, test_cwa, test_nrgs, preds, trues, seqs = evaluate(
    test_loader
)
print(
    "\nTEST RESULTS  "
    f"loss={test_loss:.4f}  acc={test_acc:.3f}  "
    f"SWA={test_swa:.3f}  CWA={test_cwa:.3f}  NRGS={test_nrgs:.3f}"
)

exp_rec["predictions"], exp_rec["ground_truth"] = preds, trues
exp_rec["metrics"]["test"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
    "cwa": test_cwa,
    "nrgs": test_nrgs,
}

# --------------------------- save artefacts --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {os.path.join(working_dir,'experiment_data.npy')}")

# quick visualisation
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(
    ["Acc", "SWA", "CWA", "NRGS"],
    [test_acc, test_swa, test_cwa, test_nrgs],
    color="skyblue",
)
ax.set_ylim(0, 1)
ax.set_title("SPR_BENCH Test Metrics")
plt.tight_layout()
plot_path = os.path.join(working_dir, "spr_metrics_bar.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
