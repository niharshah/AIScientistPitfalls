import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pathlib
from typing import List, Dict
from datasets import DatasetDict


# -------- data loading util (from prompt) ----------
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


# ---------------------------------------------------

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- load dataset --------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Data loaded:", {k: len(v) for k, v in spr.items()})


# ---------------- token + label vocab --------------
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    chars = set()
    for s in seqs:
        chars.update(list(s))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for c in sorted(chars):
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


token2id = build_vocab(spr["train"]["sequence"])
id2token = {i: t for t, i in token2id.items()}


def build_label_map(labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(labels))
    return {l: i for i, l in enumerate(uniq)}


label2id = build_label_map(spr["train"]["label"])
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print(f"Vocab size={len(token2id)}, num_labels={num_labels}")


# ------------ dataset / dataloader -----------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, token2id, label2id):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.token2id = token2id
        self.label2id = label2id

    def encode(self, seq: str) -> List[int]:
        return [self.token2id.get(ch, 1) for ch in seq]  # 1=UNK

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.encode(self.seqs[idx])
        label = self.label2id[self.labels[idx]]
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    pad_id = 0
    input_ids = []
    for x in batch:
        padded = torch.cat(
            [
                x["input_ids"],
                torch.full((max_len - len(x["input_ids"]),), pad_id, dtype=torch.long),
            ]
        )
        input_ids.append(padded)
    input_ids = torch.stack(input_ids)
    labels = torch.stack([x["label"] for x in batch])
    attn_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


train_ds = SPRDataset(spr["train"], token2id, label2id)
dev_ds = SPRDataset(spr["dev"], token2id, label2id)
test_ds = SPRDataset(spr["test"], token2id, label2id)

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


# --------------- model -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape (1,max_len,d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SPRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )  # masked mean
        return self.classifier(x)


model = SPRTransformer(len(token2id), num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------- experiment data storage -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------- training loop ---------------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, all_pred, all_true = 0.0, [], []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        all_pred.extend(torch.argmax(logits, 1).cpu().numpy())
        all_true.extend(batch["labels"].cpu().numpy())
    train_loss = total_loss / len(train_ds)
    train_f1 = f1_score(all_true, all_pred, average="macro")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(train_f1)

    # validation
    model.eval()
    val_loss, val_pred, val_true = 0.0, [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            val_pred.extend(torch.argmax(logits, 1).cpu().numpy())
            val_true.extend(batch["labels"].cpu().numpy())
    val_loss /= len(dev_ds)
    val_f1 = f1_score(val_true, val_pred, average="macro")
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val Macro_F1 = {val_f1:.4f}"
    )

# ------------- test evaluation ---------------------
model.eval()
test_pred, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        test_pred.extend(torch.argmax(logits, 1).cpu().numpy())
        test_true.extend(batch["labels"].cpu().numpy())
test_f1 = f1_score(test_true, test_pred, average="macro")
print(f"Test Macro_F1: {test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_true

# ------------- save metrics and plot ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

epochs_axis = np.arange(1, epochs + 1)
plt.figure()
plt.plot(
    epochs_axis, experiment_data["SPR_BENCH"]["losses"]["train"], label="train_loss"
)
plt.plot(epochs_axis, experiment_data["SPR_BENCH"]["losses"]["val"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve_spr.png"))

plt.figure()
plt.plot(
    epochs_axis,
    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"],
    label="train_F1",
)
plt.plot(
    epochs_axis, experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"], label="val_F1"
)
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.savefig(os.path.join(working_dir, "f1_curve_spr.png"))
