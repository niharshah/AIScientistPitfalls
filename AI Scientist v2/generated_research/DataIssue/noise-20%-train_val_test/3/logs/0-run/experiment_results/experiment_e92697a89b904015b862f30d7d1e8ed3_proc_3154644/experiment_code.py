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

import os, pathlib, time, math, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# ---------- required working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment data dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- device selection ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- dataset loader (from prompt) ----------
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


# ---------- load dataset ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- build vocabulary ----------
special_tokens = ["<PAD>"]
chars = set()
for s in spr["train"]["sequence"]:
    chars.update(list(s))
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id = stoi["<PAD>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = torch.tensor([stoi[ch] for ch in seq], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": ids, "label": label}


def collate_fn(batch):
    inputs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return {"input_ids": padded, "label": labels}


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])
test_ds = SPRTorchDataset(spr["test"])

batch_size = 128
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        masked = (~pad_mask).unsqueeze(-1).type_as(x)
        pooled = (x * masked).sum(1) / masked.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


d_model = 128
model = SimpleTransformer(
    vocab_size, d_model, nhead=4, num_layers=2, num_classes=num_classes, pad_id=pad_id
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- train / eval loops ----------
def evaluate(loader):
    model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        pad_mask = batch["input_ids"] == pad_id
        logits = model(batch["input_ids"], pad_mask)
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["label"].size(0)
    train_loss = running_loss / len(train_loader.dataset)

    val_loss, val_f1, _, _ = evaluate(dev_loader)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  validation_loss = {val_loss:.4f}  val_f1={val_f1:.4f}"
    )

    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

# ---------- final test evaluation ----------
test_loss, test_f1, test_preds, test_gts = evaluate(test_loader)
print(f"Test  : loss={test_loss:.4f}  macro_f1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# ---------- save data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
