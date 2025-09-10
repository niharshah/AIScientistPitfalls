import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from datetime import datetime

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper metrics ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def rcwa(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ---------- data loading ----------
def load_spr(root: pathlib.Path):
    try:
        from datasets import load_dataset, DatasetDict

        def _load(csv_name):
            return load_dataset(
                "csv",
                data_files=str(root / csv_name),
                split="train",
                cache_dir=".cache_dsets",
            )

        dset = {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
        print("Loaded real SPR_BENCH dataset")
    except Exception as e:
        print(f"Could not load real data ({e}); generating synthetic toy data.")

        def synth_split(n):
            data = []
            shapes = list("ABCDE")
            colors = list("abcde")
            for i in range(n):
                seq_len = random.randint(3, 8)
                seq = " ".join(
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(seq_len)
                )
                label = random.randint(0, 1)
                data.append({"id": str(i), "sequence": seq, "label": label})
            return data

        dset = {
            "train": synth_split(800),
            "dev": synth_split(200),
            "test": synth_split(200),
        }
    return dset


DATA_PATH = pathlib.Path("./SPR_BENCH")
datasets_dict = load_spr(DATA_PATH)

# ---------- vocab ----------
PAD, UNK = 0, 1


def build_vocab(seqs):
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    idx = 2
    for s in seqs:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab(
    [
        ex["sequence"] if isinstance(ex, dict) else ex["sequence"]
        for ex in datasets_dict["train"]
    ]
)


def encode(seq: str, vocab: Dict[str, int]):
    return [vocab.get(tok, UNK) for tok in seq.strip().split()]


# ---------- torch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab):
        self.seqs = [
            d["sequence"] if isinstance(d, dict) else d["sequence"] for d in data
        ]
        self.labels = [int(d["label"]) for d in data]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "ids": torch.tensor(encode(self.seqs[idx], self.vocab), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    seqs = [b["ids"] for b in batch]
    lens = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD)
    labels = torch.stack([b["label"] for b in batch])
    raw_seq = [b["raw_seq"] for b in batch]
    return {"ids": padded, "lens": lens, "label": labels, "raw_seq": raw_seq}


batch_size = 128
dl_train = DataLoader(
    SPRTorchDataset(datasets_dict["train"], vocab),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dl_val = DataLoader(
    SPRTorchDataset(datasets_dict["dev"], vocab),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
dl_test = DataLoader(
    SPRTorchDataset(datasets_dict["test"], vocab),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# ---------- model ----------
class SimpleGRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, n_classes)

    def forward(self, ids, lens):
        emb = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)  # [B, 2*hid]
        return self.fc(h)


n_classes = len(set(int(d["label"]) for d in datasets_dict["train"]))
model = SimpleGRUClassifier(len(vocab), 64, 64, n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- training ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    train_loss, n_train = 0.0, 0
    for batch in dl_train:
        ids = batch["ids"].to(device)
        lens = batch["lens"].to(device)
        lab = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(ids, lens)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * lab.size(0)
        n_train += lab.size(0)
    train_loss /= n_train
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ----- validation -----
    model.eval()
    val_loss, n_val = 0.0, 0
    all_preds, all_trues, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dl_val:
            ids = batch["ids"].to(device)
            lens = batch["lens"].to(device)
            lab = batch["label"].to(device)
            logits = model(ids, lens)
            loss = criterion(logits, lab)
            preds = logits.argmax(1).cpu().tolist()
            val_loss += loss.item() * lab.size(0)
            n_val += lab.size(0)
            all_preds.extend(preds)
            all_trues.extend(lab.cpu().tolist())
            all_seqs.extend(batch["raw_seq"])
    val_loss /= n_val
    rcwa_val = rcwa(all_seqs, all_trues, all_preds)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(rcwa_val)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | RCWA = {rcwa_val:.4f}")

# ---------- test evaluation ----------
model.eval()
all_preds, all_trues, all_seqs = [], [], []
with torch.no_grad():
    for batch in dl_test:
        ids = batch["ids"].to(device)
        lens = batch["lens"].to(device)
        lab = batch["label"].to(device)
        logits = model(ids, lens)
        preds = logits.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_trues.extend(lab.cpu().tolist())
        all_seqs.extend(batch["raw_seq"])
test_rcwa = rcwa(all_seqs, all_trues, all_preds)
print(f"\nTEST RCWA: {test_rcwa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_trues
experiment_data["SPR_BENCH"]["metrics"]["test_rcwa"] = test_rcwa
experiment_data["timestamp"] = str(datetime.now())

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
