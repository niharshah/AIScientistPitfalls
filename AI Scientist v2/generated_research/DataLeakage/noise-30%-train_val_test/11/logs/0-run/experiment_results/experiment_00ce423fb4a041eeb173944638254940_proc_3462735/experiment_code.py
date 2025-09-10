import os, math, pathlib, random, time, json
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ------------- bookkeeping -------------------------------------------------------------
experiment_data = {"dropout": {}}  # top-level key required by spec
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ------------- reproducibility helper -------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------- device -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------- data loading -----------------------------------------------------------
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
    for split in ["train", "dev", "test"]:
        dset_name = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}[split]
        dset[split] = _load(dset_name)
    return dset


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(data_root)
print({k: len(v) for k, v in spr.items()})

# ------------- tokenizer (whitespace) -------------------------------------------------
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
    ids = [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
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
        return {
            "input_ids": torch.tensor(
                encode(self.seqs[idx], max_len), dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# ------------- model ------------------------------------------------------------------
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


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
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)
        return self.classifier(x)


# ------------- training / evaluation helpers ------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    total_loss, all_preds, all_trues = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if train_flag:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train_flag:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_trues.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_trues


# ------------- hyperparameter sweep ---------------------------------------------------
dropout_vals = [0.0, 0.1, 0.2, 0.3]
num_epochs = 10

for p in dropout_vals:
    key = f"SPR_BENCH_p{p}"
    experiment_data["dropout"][key] = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    set_seed(42)  # re-seed for fair comparison
    model = CharTransformer(vocab_size, 128, 8, 2, num_labels, dropout=p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n=== Training with dropout={p} ===")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion, None)

        exp_rec = experiment_data["dropout"][key]
        exp_rec["losses"]["train"].append(tr_loss)
        exp_rec["losses"]["val"].append(val_loss)
        exp_rec["metrics"]["train_macro_f1"].append(tr_f1)
        exp_rec["metrics"]["val_macro_f1"].append(val_f1)
        exp_rec["epochs"].append(epoch)

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f} (time {time.time()-t0:.1f}s)"
        )

    # final test evaluation ------------------------------------------------------------
    test_loss, test_f1, test_preds, test_trues = run_epoch(
        model, test_loader, criterion, None
    )
    exp_rec["test_loss"] = test_loss
    exp_rec["test_macro_f1"] = test_f1
    exp_rec["predictions"] = test_preds
    exp_rec["ground_truth"] = test_trues
    print(f"Test (dropout={p}): loss={test_loss:.4f} macro_F1={test_f1:.4f}")

# ------------- save all results -------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll results saved to", os.path.join(working_dir, "experiment_data.npy"))
