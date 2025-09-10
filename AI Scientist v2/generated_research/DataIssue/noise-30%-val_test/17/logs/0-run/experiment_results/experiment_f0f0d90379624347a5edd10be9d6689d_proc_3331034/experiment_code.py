# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------#
# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ------------------------------------------------------------------#
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


spr = load_spr_bench(DATA_PATH)


# ------------------------------------------------------------------#
def build_vocab(dsets) -> dict:
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    return {ch: i + 1 for i, ch in enumerate(sorted(chars))}  # 0 reserved for PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())


def encode_sequence(seq: str, vocab: dict) -> list[int]:
    """Convert a string into list of ids (no padding)."""
    return [vocab[ch] for ch in seq]


def pad(seq_ids: list[int], L: int) -> list[int]:
    seq_ids = seq_ids[:L]
    return seq_ids + [0] * (L - len(seq_ids))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, max_len):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab, self.max_len = vocab, max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = pad(encode_sequence(self.seqs[idx], self.vocab), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# DataLoaders
batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], vocab, max_len), batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], vocab, max_len), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"], vocab, max_len), batch_size)


# ------------------------------------------------------------------#
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, mode="max"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False
        improve = (metric - self.best) if self.mode == "max" else (self.best - metric)
        if improve > self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ------------------------------------------------------------------#
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "configs": [],
    }
}


def evaluate(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            total_loss += criterion(logits, batch["labels"]).item() * batch[
                "labels"
            ].size(0)
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    loss = total_loss / len(loader.dataset)
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return loss, mcc, f1, preds, labels


def train_for_epochs(max_epochs=10, lr=1e-3, patience=3):
    model = GRUBaseline(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience=patience, mode="max")
    best_state, best_f1 = None, -1.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["labels"].size(0)
        train_loss = running_loss / len(train_loader.dataset)
        # metrics on training set
        _, train_mcc, train_f1, _, _ = evaluate(model, train_loader)

        # validation
        val_loss, val_mcc, val_f1, _, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
        )

        # log
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_f1)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
        if es(val_f1):
            print("Early stopping.")
            break

    # ----------------- Test with best checkpoint ------------------#
    model.load_state_dict(best_state)
    test_loss, test_mcc, test_f1, preds, labels = evaluate(model, test_loader)
    print(f"Test macro_F1 = {test_f1:.4f}  |  Test MCC = {test_mcc:.4f}")
    # store preds
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(labels)
    experiment_data["SPR_BENCH"]["configs"].append(
        {"epochs": max_epochs, "lr": lr, "patience": patience}
    )


# ------------------------------------------------------------------#
epoch_grid = [10, 15]  # small grid to respect runtime
lr_grid = [1e-3, 5e-4]

for ep in epoch_grid:
    for lr in lr_grid:
        print(f"\n=== Training {ep} epochs | lr={lr} ===")
        train_for_epochs(max_epochs=ep, lr=lr, patience=3)

# save everything
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
