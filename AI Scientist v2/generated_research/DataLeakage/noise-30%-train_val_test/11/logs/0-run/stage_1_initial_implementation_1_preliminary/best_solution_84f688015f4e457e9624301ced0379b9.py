import os, math, pathlib, random, time
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ----- working dir --------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ----- device -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- data loading -------------------------------------------------------------------
from datasets import load_dataset, DatasetDict


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


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(data_path)
print({k: len(v) for k, v in spr.items()})

# ----- tokeniser (whitespace-separated symbols) ---------------------------------------
PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode(seq: str, max_len: int) -> List[int]:
    tokens = seq.strip().split()
    ids = [vocab.get(t, vocab[UNK]) for t in tokens][:max_len]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = max(len(s.split()) for s in spr["train"]["sequence"])
max_len = min(max_len, 64)  # cap for speed
print(f"Sequence max_len: {max_len}")

label_set = sorted(list(set(spr["train"]["label"])))
label2id = {lab: i for i, lab in enumerate(label_set)}
num_labels = len(label2id)
print(f"Number of labels: {num_labels}")


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.seqs[idx], max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": x, "labels": y}


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# ----- model --------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, num_layers, num_labels, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


model = CharTransformer(vocab_size, 128, 8, 2, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----- training helpers ---------------------------------------------------------------
def run_epoch(loader, train_flag=True):
    if train_flag:
        model.train()
    else:
        model.eval()
    total_loss, all_preds, all_trues = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train_flag:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        trues = batch["labels"].detach().cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(trues)
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_trues


# ----- loop ---------------------------------------------------------------------------
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train_flag=True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, train_flag=False)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f} (time {time.time()-t0:.1f}s)"
    )

# ----- final test evaluation ----------------------------------------------------------
test_loss, test_f1, test_preds, test_trues = run_epoch(test_loader, train_flag=False)
print(f"Test: loss={test_loss:.4f} macro_F1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_trues
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
