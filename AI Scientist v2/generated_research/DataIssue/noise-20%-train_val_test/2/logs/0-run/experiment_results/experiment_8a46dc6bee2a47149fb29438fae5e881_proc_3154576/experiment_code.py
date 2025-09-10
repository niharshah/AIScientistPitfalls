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

import pathlib
import numpy as np
from typing import Dict, List
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ----------------------------------------------------
# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------
# Experiment data container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ----------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    """Load SPR_BENCH splits into a DatasetDict."""

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


# ----------------------------------------------------
# Paths (adjust if necessary)
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr_bench = load_spr_bench(DATA_PATH)

print({split: len(ds) for split, ds in spr_bench.items()})


# ----------------------------------------------------
# Build vocabulary from training sequences (character-level)
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    chars = set("".join(seqs))
    vocab = {pad_token: idx for idx, pad_token in enumerate(["<pad>"])}
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr_bench["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# ----------------------------------------------------
# Label mapping
labels = sorted(set(spr_bench["train"]["label"]))
label2id = {lbl: idx for idx, lbl in enumerate(labels)}
num_classes = len(label2id)
print(f"Number of classes: {num_classes}")


def encode_example(example):
    example["input_ids"] = [vocab[ch] for ch in example["sequence"]]
    example["label_id"] = label2id.get(example["label"], -1)  # -1 for test
    return example


for split in spr_bench:
    spr_bench[split] = spr_bench[split].map(
        encode_example, remove_columns=spr_bench[split].column_names
    )

spr_bench.set_format(type="python", columns=["input_ids", "label_id"])


# ----------------------------------------------------
# Collate function with dynamic padding
def collate_fn(batch):
    seqs = [item["input_ids"] for item in batch]
    labels = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
    max_len = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    attn_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn_mask[i, : len(s)] = 1
    return {
        "input_ids": padded.to(device),
        "attention_mask": attn_mask.to(device),
        "labels": labels.to(device),
    }


train_loader = DataLoader(
    spr_bench["train"], batch_size=128, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    spr_bench["dev"], batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ----------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, dim_ff=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        emb = self.embed(x)
        key_padding_mask = ~attention_mask.bool()
        enc_out = self.encoder(emb, src_key_padding_mask=key_padding_mask)
        masked = enc_out * attention_mask.unsqueeze(-1)
        pooled = masked.sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1)
        logits = self.fc(pooled)
        return logits


model = TransformerClassifier(
    vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ----------------------------------------------------
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["labels"].cpu().numpy()))
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


# ----------------------------------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_macro_f1, val_preds, val_labels = evaluate(val_loader)

    train_macro_f1, _, _, _ = evaluate(train_loader)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"train_macro_f1={train_macro_f1:.4f} | val_macro_f1={val_macro_f1:.4f}"
    )

    # store metrics
    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(train_macro_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_macro_f1)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["predictions"].append(val_preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(val_labels)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

# ----------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
