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

import os, pathlib, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# ---------- experiment data ----------
experiment_data = {"dropout_tuning": {}}
# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(file):
        return load_dataset(
            "csv", data_files=str(root / file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
num_labels = len(set(spr["train"]["label"]))
print("Loaded SPR_BENCH with", len(spr["train"]), "train examples.")


# ---------- build vocab ----------
def build_vocab(dataset):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in dataset["sequence"]:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
pad_id, unk_id = vocab["<PAD>"], vocab["<UNK>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ---------- encode sequences ----------
def encode(seq):
    return [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ---------- collate ----------
def collate_fn(batch):
    ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(x.size(0) for x in ids)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : seq.size(0)] = seq
        attn[i, : seq.size(0)] = 1
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


batch_size = 128
loaders = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
    )
    for split in ["train", "dev", "test"]
}


# ---------- model ----------
class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        nlayers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(5000, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        x = self.emb(input_ids) + self.pos_emb[:seq_len]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(x)


# ---------- training helpers ----------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- hyperparameter sweep ----------
dropout_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
epochs = 10
best_global_f1, best_dropout = 0.0, None
best_state_path = os.path.join(working_dir, "best_model_overall.pt")

for dp in dropout_values:
    tag = f"SPR_BENCH_dropout_{dp}"
    experiment_data["dropout_tuning"][tag] = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    model = CharTransformer(vocab_size, num_labels, dropout=dp).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best_val_f1 = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, loaders["train"], criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, loaders["dev"], criterion)
        ed = experiment_data["dropout_tuning"][tag]
        ed["epochs"].append(epoch)
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train_f1"].append(tr_f1)
        ed["metrics"]["val_f1"].append(val_f1)
        print(
            f"[dropout={dp}] Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(), os.path.join(working_dir, f"best_model_dp_{dp}.pt")
            )
    # evaluate on test set with best model for this dropout
    model.load_state_dict(
        torch.load(os.path.join(working_dir, f"best_model_dp_{dp}.pt"))
    )
    test_loss, test_f1, test_preds, test_gts = run_epoch(
        model, loaders["test"], criterion
    )
    experiment_data["dropout_tuning"][tag]["losses"]["test"] = test_loss
    experiment_data["dropout_tuning"][tag]["metrics"]["test_f1"] = test_f1
    experiment_data["dropout_tuning"][tag]["predictions"] = test_preds
    experiment_data["dropout_tuning"][tag]["ground_truth"] = test_gts
    print(f"[dropout={dp}] Test MacroF1: {test_f1:.4f}\n")
    if best_val_f1 > best_global_f1:
        best_global_f1 = best_val_f1
        best_dropout = dp
        torch.save(model.state_dict(), best_state_path)
    del model, optimizer, criterion
    torch.cuda.empty_cache()

print(f"Best dev MacroF1={best_global_f1:.4f} obtained with dropout={best_dropout}")

# ---------- final best model on test set ----------
best_model = CharTransformer(vocab_size, num_labels, dropout=best_dropout).to(device)
best_model.load_state_dict(torch.load(best_state_path))
criterion = nn.CrossEntropyLoss()
test_loss, test_f1, test_preds, test_gts = run_epoch(
    best_model, loaders["test"], criterion
)
print(f"Final Test MacroF1 with best dropout ({best_dropout}): {test_f1:.4f}")

# save final best predictions separately
experiment_data["best_overall"] = {
    "dropout": best_dropout,
    "test_f1": test_f1,
    "predictions": test_preds,
    "ground_truth": test_gts,
}

# ---------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
