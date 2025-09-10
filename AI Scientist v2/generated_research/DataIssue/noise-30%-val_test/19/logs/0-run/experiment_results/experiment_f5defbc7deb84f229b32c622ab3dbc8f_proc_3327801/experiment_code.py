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
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
from typing import List, Dict

# ---------------- GPU / device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- data loading ----------------
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


def get_dataset() -> DatasetDict:
    possible = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for p in possible:
        if (p / "train.csv").exists():
            print(f"Loading real SPR_BENCH from {p}")
            return load_spr_bench(p)
    # --------- synthetic fallback ------------
    print("SPR_BENCH not found, creating synthetic toy dataset")

    def synth(n):
        rows = []
        shapes = "ABCD"
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            label = int(seq.count("A") % 2 == 0)  # even # of A -> label 1
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    dset = DatasetDict()
    dset["train"] = to_ds(synth(2000))
    dset["dev"] = to_ds(synth(500))
    dset["test"] = to_ds(synth(500))
    return dset


spr = get_dataset()

# -------------- vocabulary -------------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(list(set(all_text)))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 is PAD
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
print(f"Vocab size: {vocab_size-1}")

max_len = min(100, max(len(s) for s in spr["train"]["sequence"]))  # cap at 100


def encode(seq: str) -> List[int]:
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.y = hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.seq[idx]), dtype=torch.long)
        y = torch.tensor(int(self.y[idx]), dtype=torch.float)
        return {"input_ids": x, "label": y}


batch_size = 128
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size)


# -------------- model ------------------------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int = 64, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.rnn(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        logits = self.fc(h).squeeze(1)
        return logits


model = CharBiGRU(vocab_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------- tracking ---------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# -------------- training loop ---------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    tr_losses = []
    tr_outputs = []
    tr_labels = []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        tr_outputs.extend(preds)
        tr_labels.extend(batch["label"].long().cpu().numpy())
    train_f1 = f1_score(tr_labels, tr_outputs, average="macro")
    val_losses = []
    val_outputs = []
    val_labels = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            val_losses.append(loss.item())
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            val_outputs.extend(preds)
            val_labels.extend(batch["label"].long().cpu().numpy())
    val_f1 = f1_score(val_labels, val_outputs, average="macro")

    print(
        f"Epoch {epoch}: validation_loss = {np.mean(val_losses):.4f}, val_macro_f1 = {val_f1:.4f}"
    )

    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(np.mean(tr_losses))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(np.mean(val_losses))
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

# -------------- test evaluation --------------
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(batch["label"].long().cpu().numpy())
test_macro_f1 = f1_score(test_labels, test_preds, average="macro")
print(f"Test Macro-F1: {test_macro_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# -------------- visualization ----------------
plt.figure(figsize=(6, 4))
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["train"],
    label="train_loss",
)
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["val"],
    label="val_loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("SPR_BENCH_loss_curve")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
plt.close()
