import os, math, pathlib, random, time
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ------------------------------------------------------------------ paths / bookkeeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH_hybrid": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------ device & seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(2024)

# ------------------------------------------------------------------ data
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ------------------------------- vocab & encoding (whitespace tokenizer)
PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
vocab_size = len(vocab)
print("vocab_size", vocab_size)


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in seq.split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
num_labels = len(label2id)


# ------------------------------- dataset
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labs = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def _symbolic_feats(self, seq_tokens):
        L = len(seq_tokens)
        uniq = len(set(seq_tokens))
        rep = (L - uniq) / L if L > 0 else 0
        return [L / max_len, uniq / max_len, rep]  # simple scale 0-1

    def __getitem__(self, idx):
        tokens = self.seqs[idx].split()
        return {
            "input_ids": torch.tensor(
                encode(self.seqs[idx], max_len), dtype=torch.long
            ),
            "sym_feats": torch.tensor(self._symbolic_feats(tokens), dtype=torch.float),
            "labels": torch.tensor(self.labs[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# ------------------------------- model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridTransformer(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, nhead, n_layers, num_labels, sym_dim=3, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Linear(emb_dim + sym_dim, num_labels)

    def forward(self, input_ids, sym_feats):
        mask = input_ids == 0
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0).mean(1)  # mean pool
        x = torch.cat([x, sym_feats], dim=-1)
        return self.cls(x)


model = HybridTransformer(vocab_size, 128, 8, 2, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------- train / eval
def run_epoch(loader, train_flag=True):
    model.train() if train_flag else model.eval()
    tot_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if train_flag:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"], batch["sym_feats"])
            loss = criterion(logits, batch["labels"])
            if train_flag:
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    return (
        tot_loss / len(loader.dataset),
        f1_score(trues, preds, average="macro"),
        preds,
        trues,
    )


# ------------------------------- training loop
num_epochs = 8
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, False)

    experiment_data["SPR_BENCH_hybrid"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH_hybrid"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH_hybrid"]["metrics"]["train_macro_f1"].append(tr_f1)
    experiment_data["SPR_BENCH_hybrid"]["metrics"]["val_macro_f1"].append(val_f1)
    experiment_data["SPR_BENCH_hybrid"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f}  ({time.time()-t0:.1f}s)"
    )

# ------------------------------- final test
test_loss, test_f1, test_preds, test_trues = run_epoch(test_loader, False)
experiment_data["SPR_BENCH_hybrid"]["test_macro_f1"] = test_f1
experiment_data["SPR_BENCH_hybrid"]["test_loss"] = test_loss
experiment_data["SPR_BENCH_hybrid"]["predictions"] = test_preds
experiment_data["SPR_BENCH_hybrid"]["ground_truth"] = test_trues
print(f"Test: loss={test_loss:.4f} macro_F1={test_f1:.4f}")

# ------------------------------- save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
