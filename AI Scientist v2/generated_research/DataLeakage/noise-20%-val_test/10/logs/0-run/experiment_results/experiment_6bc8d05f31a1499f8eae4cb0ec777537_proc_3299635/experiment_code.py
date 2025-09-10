import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pathlib
import json
from typing import List, Dict

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- data loading -----------------
# helper copied from the prompt (no pandas)
from datasets import load_dataset


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


# try default location or relative fallback
default_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not default_path.exists():
    default_path = pathlib.Path("SPR_BENCH/")
spr = load_spr_bench(default_path)
print({k: len(v) for k, v in spr.items()})


# ---------------- vocabulary -----------------
def build_vocab(dataset) -> Dict[str, int]:
    charset = set()
    for seq in dataset["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(list(charset)))}  # 0 is PAD
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(spr["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ---------------- dataset -----------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seq = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seq)

    def encode(self, s: str) -> List[int]:
        return [self.vocab[c] for c in s]

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    inputs = [item["input"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = [len(x) for x in inputs]
    max_len = max(lengths)
    padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, seq in enumerate(inputs):
        padded[i, : len(seq)] = seq
    return {"input": padded, "lengths": torch.tensor(lengths), "label": labels}


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"], vocab),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_dl = DataLoader(
    SPRTorchDataset(spr["dev"], vocab),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_dl = DataLoader(
    SPRTorchDataset(spr["test"], vocab),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

num_classes = len(set(spr["train"]["label"]))
print("Classes:", num_classes)


# ---------------- model -----------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # mean pooling over valid timesteps
        mask = (x != 0).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lens = lengths.unsqueeze(1).to(out.dtype)
        mean = summed / lens
        logits = self.fc(mean)
        return logits


model = CharBiLSTM(vocab_size, emb_dim=64, hidden_dim=128, num_classes=num_classes).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------- experiment storage -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------- helpers -----------------
def run_epoch(dl, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().numpy())
    avg_loss = total_loss / len(dl.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


# ---------------- training loop -----------------
epochs = 10
best_dev_f1 = 0.0
for epoch in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train=True)
    val_loss, val_f1, _, _ = run_epoch(dev_dl, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"train_f1={tr_f1:.4f} val_f1={val_f1:.4f}"
    )
    if val_f1 > best_dev_f1:
        best_dev_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))

# ---------------- evaluation on test -----------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best_model.pt")))
test_loss, test_f1, preds, gts = run_epoch(test_dl, train=False)
print(f"Test Macro_F1_Score: {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = gts.tolist()

# ---------------- save metrics -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------------- visualization -----------------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["metrics"]["val_f1"], label="Val Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("Validation Macro-F1 over epochs")
plt.legend()
plt.savefig(os.path.join(working_dir, "spr_val_f1_curve.png"))
plt.close()
