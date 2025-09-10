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
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ----------- device -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------- data loader utilities (given) -----------
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


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def harmonic_weighted_accuracy(sequences, y_true, y_pred):
    swa = shape_weighted_accuracy(sequences, y_true, y_pred)
    cwa = color_weighted_accuracy(sequences, y_true, y_pred)
    return 0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)


# ----------- simple vocabulary -----------
class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.stoi[tok] for tok in tokens]


# ----------- model -----------
class BagClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# ----------- data path -----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ----------- build vocab and label map -----------
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# ----------- collate function -----------
def collate_batch(batch):
    token_ids = []
    offsets = [0]
    label_ids = []
    for ex in batch:
        seq, lab = ex["sequence"], ex["label"]
        ids = vocab(seq.split())
        token_ids.extend(ids)
        offsets.append(offsets[-1] + len(ids))
        label_ids.append(label2id[lab])
    offsets = torch.tensor(offsets[:-1], dtype=torch.long)
    text = torch.tensor(token_ids, dtype=torch.long)
    labels_t = torch.tensor(label_ids, dtype=torch.long)
    return text.to(device), offsets.to(device), labels_t.to(device)


# ----------- dataloaders -----------
batch_size = 128
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)

# ----------- experiment tracking -----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ----------- training setup -----------
embed_dim = 64
model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5


# ----------- helpers -----------
def evaluate(data_loader):
    model.eval()
    y_true, y_pred, sequences = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (text, offsets, labels_t) in enumerate(data_loader):
            outputs = model(text, offsets)
            loss = criterion(outputs, labels_t)
            total_loss += loss.item() * labels_t.size(0)
            preds = outputs.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in labels_t.cpu().tolist()])
            # recover sequences for metric weighting
            start = batch_idx * batch_size
            end = start + labels_t.size(0)
            sequences.extend(data_loader.dataset["sequence"][start:end])
    avg_loss = total_loss / len(y_true)
    swa = shape_weighted_accuracy(sequences, y_true, y_pred)
    cwa = color_weighted_accuracy(sequences, y_true, y_pred)
    hwa = harmonic_weighted_accuracy(sequences, y_true, y_pred)
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# ----------- training loop -----------
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for text, offsets, labels_t in train_loader:
        optimizer.zero_grad()
        outputs = model(text, offsets)
        loss = criterion(outputs, labels_t)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels_t.size(0)
    train_loss = running_loss / len(spr["train"])
    val_loss, swa, cwa, hwa, _, _ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"SWA": swa, "CWA": cwa, "HWA": hwa}
    )

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | HWA={hwa:.4f}"
    )

# ----------- final test evaluation -----------
test_loss, swa_t, cwa_t, hwa_t, y_true_t, y_pred_t = evaluate(test_loader)
print(
    f"Test: loss={test_loss:.4f} | SWA={swa_t:.4f} | CWA={cwa_t:.4f} | HWA={hwa_t:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = y_pred_t
experiment_data["SPR_BENCH"]["ground_truth"] = y_true_t
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
