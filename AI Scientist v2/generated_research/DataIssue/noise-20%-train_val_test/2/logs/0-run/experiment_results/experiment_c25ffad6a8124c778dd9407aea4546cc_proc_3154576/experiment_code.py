import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment data container --------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------- dataset loader utility (from prompt) --------------------
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


# -------------------- PyTorch dataset --------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        ids = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in seq[: self.max_len]]
        length = len(ids)
        if length < self.max_len:
            ids += [self.pad_id] * (self.max_len - length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# -------------------- model --------------------
class SPRModel(nn.Module):
    def __init__(
        self, vocab_size, num_classes, d_model=128, nhead=4, num_layers=2, max_len=128
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, : x.size(1), :]
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # (batch, seq, d_model)
        x = x.mean(dim=1)
        return self.cls(x)


# -------------------- training utils --------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, all_pred, all_true = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds = out.argmax(dim=1).detach().cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    return avg_loss, macro_f1


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_pred, all_true = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        total_loss += loss.item() * batch["labels"].size(0)
        preds = out.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    return avg_loss, macro_f1, all_pred, all_true


# -------------------- main routine --------------------
def main():
    # Path to SPR_BENCH folder (adjust if needed)
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset path {DATA_PATH} not found.")
    spr = load_spr_bench(DATA_PATH)

    # Build vocab from training sequences
    chars = set("".join(spr["train"]["sequence"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    max_len = min(128, max(len(s) for s in spr["train"]["sequence"]))

    train_ds = SPRTorchDataset(spr["train"], vocab, max_len)
    val_ds = SPRTorchDataset(spr["dev"], vocab, max_len)
    test_ds = SPRTorchDataset(spr["test"], vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    num_classes = len(set(spr["train"]["label"]))
    model = SPRModel(len(vocab), num_classes, max_len=max_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion)
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
        )

        experiment_data["SPR_BENCH"]["epochs"].append(epoch)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(tr_f1)
        experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)

    # Final test evaluation
    test_loss, test_f1, preds, gts = eval_epoch(model, test_loader, criterion)
    print(f"Test macro_f1 = {test_f1:.4f}")
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts

    # Save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Execute immediately
main()
