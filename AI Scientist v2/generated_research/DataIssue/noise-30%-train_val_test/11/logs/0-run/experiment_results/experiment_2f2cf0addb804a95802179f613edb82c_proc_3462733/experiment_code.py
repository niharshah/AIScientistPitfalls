#!/usr/bin/env python
import os, math, pathlib, random, time, json
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# -------------------- experiment dict -----------------------------
experiment_data = {"emb_dim_tuning": {"SPR_BENCH": {}}}

# -------------------- reproducibility -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------- device --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- data loading --------------------------------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


data_root = pathlib.Path(
    os.getenv("SPR_DATA_DIR", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(data_root)
print({k: len(v) for k, v in spr.items()})

# -------------------- tokeniser -----------------------------------
PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode(seq: str, max_len: int) -> List[int]:
    tokens = seq.strip().split()
    ids = [vocab.get(t, vocab[UNK]) for t in tokens][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
print(f"Sequence max_len: {max_len}")

label_set = sorted(list(set(spr["train"]["label"])))
label2id = {lab: i for i, lab in enumerate(label_set)}
num_labels = len(label2id)
print(f"Number of labels: {num_labels}")


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode(self.seqs[idx], max_len), dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# -------------------- model defs ----------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, num_layers, num_labels, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)
        return self.classifier(x)


def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    total_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if train_flag:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train_flag:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(trues, preds, average="macro")
    return avg_loss, macro_f1, preds, trues


# -------------------- tuning loop -------------------------------
emb_dims = [128, 192, 256, 384]
num_epochs = 8
for emb_dim in emb_dims:
    print(f"\n===== Training with emb_dim = {emb_dim} =====")
    model = CharTransformer(vocab_size, emb_dim, 8, 2, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_log = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)
        run_log["losses"]["train"].append(tr_loss)
        run_log["losses"]["val"].append(val_loss)
        run_log["metrics"]["train_macro_f1"].append(tr_f1)
        run_log["metrics"]["val_macro_f1"].append(val_f1)
        run_log["epochs"].append(epoch)
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    # final test evaluation
    test_loss, test_f1, test_preds, test_trues = run_epoch(
        model, test_loader, criterion
    )
    print(f"Test emb_dim={emb_dim}: loss={test_loss:.4f} macro_F1={test_f1:.4f}")

    run_log["test_loss"] = test_loss
    run_log["test_macro_f1"] = test_f1
    run_log["predictions"] = test_preds
    run_log["ground_truth"] = test_trues

    experiment_data["emb_dim_tuning"]["SPR_BENCH"][f"emb_{emb_dim}"] = run_log

# -------------------- save ---------------------------------------
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
