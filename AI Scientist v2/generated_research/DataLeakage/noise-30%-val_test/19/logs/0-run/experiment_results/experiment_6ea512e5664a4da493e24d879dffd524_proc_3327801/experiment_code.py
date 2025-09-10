import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------ Experiment data dict -------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# --------------- Dataset loading ---------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr)

# -------------- Vocabulary -----------------
all_chars = set()
for ex in spr["train"]["sequence"]:
    all_chars.update(list(ex))
PAD, UNK = 0, 1
itos = ["<pad>", "<unk>"] + sorted(all_chars)
stoi = {c: i for i, c in enumerate(itos)}
vocab_size = len(itos)
print("Vocab size:", vocab_size)


def encode(seq):
    return [stoi.get(ch, UNK) for ch in seq]


max_len = max(len(s) for s in spr["train"]["sequence"])


# -------------- PyTorch Dataset -------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [int(l) for l in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_pad(batch):
    lengths = [len(b["input_ids"]) for b in batch]
    maxl = max(lengths)
    input_ids = torch.full((len(batch), maxl), PAD, dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "label": labels}


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])
test_ds = SPRTorchDataset(spr["test"])

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_pad)
dev_dl = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_pad)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pad)


# ---------------- Model ---------------------
class CharGRU(nn.Module):
    def __init__(self, vocab, emb=32, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        logits = self.fc(h.squeeze(0))
        return logits


model = CharGRU(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5


# ---------------- Training loop -------------
def run_epoch(dl, train_flag=False):
    if train_flag:
        model.train()
    else:
        model.eval()
    total_loss, preds, golds = 0.0, [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            if train_flag:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["label"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        golds.extend(batch["label"].cpu().tolist())
    avg_loss = total_loss / len(dl.dataset)
    f1 = f1_score(golds, preds, average="macro")
    return avg_loss, f1, preds, golds


for epoch in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train_flag=True)
    val_loss, val_f1, _, _ = run_epoch(dev_dl, train_flag=False)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

# -------------- Final test evaluation --------------
test_loss, test_f1, test_preds, test_golds = run_epoch(test_dl, train_flag=False)
print(f"Test macro_f1 = {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_golds
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_f1"] = test_f1

# -------------- Save everything --------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
