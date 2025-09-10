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

import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
# working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# helper to load SPR benchmark (copied from prompt)
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


# -------------------------------------------------
# attempt to load dataset, otherwise create synthetic
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_data = DATA_PATH.exists()
if have_data:
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH not found, generating synthetic dataset.")

    def synth_split(n):
        rows = []
        for i in range(n):
            seq_len = random.randint(5, 15)
            seq = "".join(random.choices(string.ascii_uppercase[:10], k=seq_len))
            label = int(seq.count("A") % 2 == 0)  # simple parity rule
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def to_hf(rows):
        return DatasetDict(
            {"train": load_dataset("json", data_files={"train": [rows]}, split="train")}
        )["train"]

    spr = DatasetDict()
    spr["train"] = load_dataset(
        "json", data_files={"train": synth_split(2000)}, split="train"
    )
    spr["dev"] = load_dataset(
        "json", data_files={"train": synth_split(400)}, split="train"
    )
    spr["test"] = load_dataset(
        "json", data_files={"train": synth_split(400)}, split="train"
    )

print({k: len(v) for k, v in spr.items()})

# -------------------------------------------------
# build vocabulary
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq, max_len):
    ids = [vocab.get(ch, 1) for ch in seq][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


max_len = max(len(ex["sequence"]) for ex in spr["train"])
max_len = min(max_len, 120)


# -------------------------------------------------
# PyTorch Dataset wrapper
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(
                encode(ex["sequence"], max_len), dtype=torch.long
            ),
            "label": torch.tensor(int(ex["label"]), dtype=torch.long),
        }


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# -------------------------------------------------
# Model
class CharGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        logits = self.fc(h.squeeze(0))
        return logits


model = CharGRU(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------
# tracking dict
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------------------------------------------
# training loop
epochs = 5
for epoch in range(1, epochs + 1):
    # train
    model.train()
    total_loss, total_items = 0.0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        total_items += batch["labels"].size(0)
    train_loss = total_loss / total_items
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # validation
    model.eval()
    val_loss, val_items = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            val_items += batch["labels"].size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(list(preds))
            all_labels.extend(list(labels))
    val_loss /= val_items
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(macro_f1)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, Macro-F1 = {macro_f1:.4f}")

# Store predictions and ground truth from final epoch
experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_labels

# -------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
