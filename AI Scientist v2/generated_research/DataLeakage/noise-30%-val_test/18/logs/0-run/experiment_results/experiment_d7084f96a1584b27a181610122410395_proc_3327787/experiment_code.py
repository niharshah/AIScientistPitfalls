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

import pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# --------------- device handling ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------- load SPR_BENCH -------------------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if data_path.exists():
    dsets = load_spr_bench(data_path)
else:
    # fallback synthetic data (very small, just to guarantee runnable script)
    from datasets import Dataset, DatasetDict

    def synth_split(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 15)
            seq = "".join(
                random.choices(list(string.ascii_lowercase) + ["#", "@", "&"], k=L)
            )
            lbl = int(seq.count("#") % 2 == 0)  # arbitrary rule
            seqs.append(seq)
            labels.append(lbl)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict()
    dsets["train"] = synth_split(512)
    dsets["dev"] = synth_split(128)
    dsets["test"] = synth_split(128)
print({k: len(v) for k, v in dsets.items()})

# --------------- vocab & encoding -----------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab.get(ch, vocab[UNK]) for ch in seq]


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# --------------- Dataset / Dataloader -------------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": padded, "labels": labels}


batch_size = 128
train_loader = DataLoader(
    dsets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    dsets["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    dsets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)


# --------------- Model ----------------------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        em = self.emb(x)
        packed_out, _ = self.lstm(em)
        # simple max-pool over time
        pooled, _ = torch.max(packed_out, dim=1)
        return self.fc(pooled)


model = SPRClassifier(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------- experiment_data dict -------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# --------------- training loop --------------------
epochs = 5
best_f1 = 0.0
for epoch in range(1, epochs + 1):
    model.train()
    train_losses, train_preds, train_gts = [], [], []
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_preds.extend(logits.argmax(1).cpu().numpy())
        train_gts.extend(batch["labels"].cpu().numpy())
    train_f1 = f1_score(train_gts, train_preds, average="macro")
    # ---- eval on dev ----
    model.eval()
    dev_losses, dev_preds, dev_gts = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            dev_losses.append(loss.item())
            dev_preds.extend(logits.argmax(1).cpu().numpy())
            dev_gts.extend(batch["labels"].cpu().numpy())
    dev_f1 = f1_score(dev_gts, dev_preds, average="macro")
    print(
        f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_loss={np.mean(dev_losses):.4f}, val_macroF1={dev_f1:.4f}"
    )
    # store experiment data
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(dev_f1)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(np.mean(train_losses))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(np.mean(dev_losses))
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    # save best preds
    if dev_f1 > best_f1:
        best_f1 = dev_f1
        experiment_data["SPR_BENCH"]["predictions"] = dev_preds
        experiment_data["SPR_BENCH"]["ground_truth"] = dev_gts

# --------------- final test evaluation ------------
model.eval()
test_preds, test_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_gts.extend(batch["labels"].cpu().numpy())
test_f1 = f1_score(test_gts, test_preds, average="macro")
print(f"Best Dev Macro-F1 = {best_f1:.4f} | Test Macro-F1 = {test_f1:.4f}")

# --------------- save experiment data -------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
