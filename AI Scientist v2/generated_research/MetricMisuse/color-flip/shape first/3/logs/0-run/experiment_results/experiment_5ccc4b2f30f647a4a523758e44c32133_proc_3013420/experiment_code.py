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

import os, random, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# working dir for artifacts ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device ----------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------------------------------------
# Utility functions (adapted from provided SPR.py)
# ------------------------------------------------------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) + 1e-9)


def try_load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset, DatasetDict

        def _ld(split_csv):
            return load_dataset(
                "csv",
                data_files=str(root / split_csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = {}
        for sp in ["train.csv", "dev.csv", "test.csv"]:
            d[sp.split(".")[0]] = _ld(sp)
        return True, d
    except Exception as e:
        print("Could not load SPR_BENCH, falling back to synthetic data.", e)
        return False, {}


# Synthetic data -------------------------------------------------------------------------------
def make_synth_dataset(n_rows):
    shapes = list("ABCDE")
    colors = list("12345")
    sequences, labels = [], []
    for _ in range(n_rows):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        sequences.append(seq)
        # simple rule: label = 1 if #unique shapes > #unique colors else 0
        labels.append(int(count_shape_variety(seq) > count_color_variety(seq)))
    return {"sequence": sequences, "label": labels}


# Dataset wrapper -------------------------------------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_len):
        self.seqs, self.labels = sequences, labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self.encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


# Model -----------------------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        logits = self.fc(h.squeeze(0))
        return logits


# Data preparation ------------------------------------------------------------------------------
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw_dsets = try_load_spr_bench(SPR_PATH)

if have_real:
    train_rows = raw_dsets["train"]
    dev_rows = raw_dsets["dev"]
    test_rows = raw_dsets["test"]
    train_dict = {"sequence": train_rows["sequence"], "label": train_rows["label"]}
    dev_dict = {"sequence": dev_rows["sequence"], "label": dev_rows["label"]}
    test_dict = {"sequence": test_rows["sequence"], "label": test_rows["label"]}
else:
    train_dict = make_synth_dataset(2000)
    dev_dict = make_synth_dataset(400)
    test_dict = make_synth_dataset(400)

all_tokens = set(tok for seq in train_dict["sequence"] for tok in seq.split())
vocab = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
vocab["<pad>"] = 0
vocab["<unk>"] = 1
pad_idx = vocab["<pad>"]
max_len = max(len(s.split()) for s in train_dict["sequence"])

train_ds = SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len)
dev_ds = SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len)
test_ds = SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# Experiment bookkeeping ------------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# Instantiate model -----------------------------------------------------------------------------
num_classes = len(set(train_dict["label"]))
model = GRUClassifier(
    len(vocab), emb_dim=64, hid_dim=128, num_classes=num_classes, pad_idx=pad_idx
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop ---------------------------------------------------------------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n = 0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["y"].size(0)
        n += batch["y"].size(0)
    train_loss = total_loss / n
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # Validation -------------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        val_loss, n = 0, 0
        all_pred, all_true, all_seq = [], [], []
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            val_loss += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(batch["y"].cpu().tolist())
            all_seq.extend(batch["raw"])
        val_loss /= n
        val_scwa = scwa(all_seq, all_true, all_pred)

    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_scwa)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SCWA = {val_scwa:.4f}"
    )

# Final test evaluation -------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    all_pred, all_true, all_seq = [], [], []
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["x"])
        preds = logits.argmax(1).cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(batch["y"].cpu().tolist())
        all_seq.extend(batch["raw"])
    test_scwa = scwa(all_seq, all_true, all_pred)
    print(f"Test SCWA = {test_scwa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = all_pred
experiment_data["SPR_BENCH"]["ground_truth"] = all_true

# Save everything -------------------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
