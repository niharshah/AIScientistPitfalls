import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import random
import time
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

# ---------------- Device -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Utility: load dataset or create toy one --------------------------------
def load_real_or_synthetic() -> DatasetDict:
    """
    Try to load real SPR_BENCH data; otherwise return a tiny synthetic set.
    """
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        from SPR import (
            load_spr_bench,
        )  # provided helper script assumed to be importable

        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH from", DATA_PATH)
    except Exception as e:
        print("Real SPR_BENCH not found, generating synthetic toy set.", e)

        # simple synthetic generator
        def gen(n):
            seqs, labs = [], []
            shapes = ["A", "B", "C"]
            colors = ["r", "g", "b"]
            for _ in range(n):
                ln = random.randint(3, 6)
                tokens = [
                    random.choice(shapes) + random.choice(colors) for _ in range(ln)
                ]
                seq = " ".join(tokens)
                # toy rule: label 1 if more distinct shapes than colors else 0
                s_var = len(set(t[0] for t in tokens))
                c_var = len(set(t[1] for t in tokens))
                label = 1 if s_var > c_var else 0
                seqs.append(seq)
                labs.append(label)
            return {"sequence": seqs, "label": labs}

        train_ds = gen(200)
        dev_ds = gen(50)
        test_ds = gen(50)
        dset = DatasetDict()
        dset["train"] = (
            torch.utils.data.TensorDataset()
        )  # placeholder to satisfy type checker
        from datasets import Dataset

        dset["train"] = Dataset.from_dict(train_ds)
        dset["dev"] = Dataset.from_dict(dev_ds)
        dset["test"] = Dataset.from_dict(test_ds)
    return dset


dset = load_real_or_synthetic()


# ---------------- Tokenisation -----------------------------------------------------------
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for s in seqs:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
print("Vocab size:", len(vocab))


def encode_sequence(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<UNK>"]) for tok in seq.strip().split()]


max_len = max(len(seq.split()) for seq in dset["train"]["sequence"])


# ---------------- Dataset / Dataloader ---------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seqs = sequences
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {"sequence": self.seqs[idx], "label": self.labels[idx]}


def collate_fn(batch):
    seqs = [encode_sequence(item["sequence"], vocab) for item in batch]
    lens = [len(s) for s in seqs]
    max_l = max(lens)
    padded = [s + [0] * (max_l - len(s)) for s in seqs]
    input_ids = torch.tensor(padded, dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "raw_seq": [item["sequence"] for item in batch],
    }


batch_size = 64
train_loader = DataLoader(
    SPRDataset(dset["train"]["sequence"], dset["train"]["label"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_loader = DataLoader(
    SPRDataset(dset["dev"]["sequence"], dset["dev"]["label"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

num_classes = len(set(dset["train"]["label"]))


# ---------------- Model ------------------------------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lin = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        emb = self.emb(input_ids)  # (B,L,E)
        mask = (input_ids != 0).unsqueeze(-1)  # (B,L,1)
        summed = (emb * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        mean = summed / lengths  # (B,E)
        return self.lin(mean)


model = MeanPoolClassifier(len(vocab), 32, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------- Metrics ----------------------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def crwa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ---------------- Experiment data structure ---------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_crwa": [], "val_crwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------------- Training loop ----------------------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- Evaluation on dev
    model.eval()
    val_loss = 0.0
    all_preds, all_trues, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1).cpu().tolist()
            trues = batch["labels"].cpu().tolist()
            seqs = batch["raw_seq"]
            all_preds.extend(preds)
            all_trues.extend(trues)
            all_seqs.extend(seqs)
    val_loss /= len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    # metrics
    train_crwa = None  # compute quickly on subset if desired; skipping for brevity
    val_crwa = crwa(all_seqs, all_trues, all_preds)
    experiment_data["SPR_BENCH"]["metrics"]["val_crwa"].append(val_crwa)
    experiment_data["SPR_BENCH"]["predictions"].append(all_preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(all_trues)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_CRWA = {val_crwa:.4f}")

# ---------------- Save experiment data ---------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
