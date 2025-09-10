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

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ------------------ reproducibility ------------------ #
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ------------------ device --------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ load SPR_BENCH ------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)


# ------------------ vocabulary ----------------------- #
def build_vocab(dsets):
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    return {ch: i + 1 for i, ch in enumerate(sorted(chars))}  # 0=PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1


def encode(seq: str):
    return [vocab[c] for c in seq]


# ------------------ dataset & collate ---------------- #
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    ids = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
    max_len = max(len(x) for x in ids)
    padded = torch.zeros(len(ids), max_len, dtype=torch.long)
    attn_mask = torch.zeros_like(padded)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = seq
        attn_mask[i, : len(seq)] = 1
    return {
        "input_ids": padded,
        "attention_mask": attn_mask,
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


batch_size = 256
train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(SPRDataset(spr["dev"]), batch_size, collate_fn=collate)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size, collate_fn=collate)


# ------------- model: transformer encoder ------------ #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # 1 x max_len x d_model

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerSPR(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        key_padding = attention_mask == 0
        enc = self.encoder(x, src_key_padding_mask=key_padding)
        masked_enc = enc * attention_mask.unsqueeze(-1)
        pooled = masked_enc.sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
        return self.fc(pooled).squeeze(1)


# ------------------- training utils ------------------ #
def evaluate(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return (
        total_loss / len(loader.dataset),
        matthews_corrcoef(labels, preds),
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


class EarlyStop:
    def __init__(self, patience=4, delta=1e-4):
        self.best = None
        self.patience = patience
        self.delta = delta
        self.count = 0

    def step(self, metric):
        if self.best is None or metric > self.best + self.delta:
            self.best = metric
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


# -------------------- experiment log ----------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------- training loop ------------------ #
def train_model(epochs=12, lr=1e-3):
    model = TransformerSPR(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()
    early = EarlyStop(patience=3)
    best_state, best_mcc = None, -1
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * batch["labels"].size(0)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)
        _, train_mcc, train_f1, _, _ = evaluate(model, train_loader)
        val_loss, val_mcc, val_f1, _, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_MCC = {val_mcc:.4f}"
        )
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_mcc)
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_state = model.state_dict()
        if early.step(val_mcc):
            print("Early stopping")
            break
    model.load_state_dict(best_state)
    tloss, tmcc, tf1, preds, labels = evaluate(model, test_loader)
    print(f"Test MCC = {tmcc:.4f} | Test Macro-F1 = {tf1:.4f}")
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(labels)


train_model(epochs=12, lr=1e-3)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
