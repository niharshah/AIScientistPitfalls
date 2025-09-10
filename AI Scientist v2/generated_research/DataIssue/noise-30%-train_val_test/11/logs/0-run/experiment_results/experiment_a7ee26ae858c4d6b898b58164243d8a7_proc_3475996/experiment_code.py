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

# No-PadMask Transformer ablation – complete, runnable script
import os, math, random, time, pathlib, json, csv
from typing import List, Dict
import numpy as np

# ------------------------------------------------- working dir & meta
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "NoPadMask_Transformer": {
        "SPR_BENCH": {
            "metrics": {"train_macro_f1": [], "val_macro_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}


# ------------------------------------------------- reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    import torch, os

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ------------------------------------------------- optional synthetic data generation
def generate_synthetic_spr(root: pathlib.Path, n_train=2000, n_dev=500, n_test=700):
    root.mkdir(parents=True, exist_ok=True)
    tokens = [chr(i) for i in range(65, 91)]  # A-Z

    def make_row(idx: int):
        length = random.randint(5, 12)
        seq_toks = random.choices(tokens, k=length)
        label = "evenA" if seq_toks.count("A") % 2 == 0 else "oddA"
        return (idx, " ".join(seq_toks), label)

    def dump(split, n):
        with (root / f"{split}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                w.writerow(make_row(i))

    dump("train", n_train)
    dump("dev", n_dev)
    dump("test", n_test)


# ------------------------------------------------- ensure dataset exists
DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists() or not all(
    (DATA_PATH / f).exists() for f in ["train.csv", "dev.csv", "test.csv"]
):
    print("SPR_BENCH not found – generating synthetic data.")
    generate_synthetic_spr(DATA_PATH)
else:
    print("SPR_BENCH found – using existing files.")

# ------------------------------------------------- torch & device
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------- load dataset with 🤗 Datasets
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path):
    def _load(csvn):
        return load_dataset(
            "csv", data_files=str(root / csvn), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            sp: _load(csvn)
            for sp, csvn in zip(
                ["train", "dev", "test"], ["train.csv", "dev.csv", "test.csv"]
            )
        }
    )


spr = load_spr_bench(DATA_PATH)
print("Split sizes:", {k: len(v) for k, v in spr.items()})

# ------------------------------------------------- vocab & encoding
PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs: List[str]):
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
vocab_size = len(vocab)
max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
labels = sorted(list(set(spr["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_labels = len(labels)


def encode(seq: str):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# ------------------------------------------------- model definitions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NoPadMaskTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, layers, num_labels, dropout=0.1):
        super().__init__()
        # NO padding_idx so PAD tokens get trainable embeddings
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.proj_rel = nn.Linear(emb_dim, emb_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_labels),
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x)  # NO src_key_padding_mask
        pooled = x.mean(1)  # mean over ALL positions, pads included
        proj = self.proj_rel(x)
        scores = torch.relu(torch.matmul(proj, x.transpose(1, 2)))
        rel_vec = torch.bmm(scores.softmax(-1), x).mean(1)
        fused = torch.cat([pooled, rel_vec], dim=-1)
        return self.classifier(fused)


# ------------------------------------------------- helpers
def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    tot_loss, all_preds, all_trues = 0.0, [], []
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
        bs = batch["labels"].size(0)
        tot_loss += loss.item() * bs
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_trues.extend(batch["labels"].cpu().numpy())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_trues


# ------------------------------------------------- training loop
set_seed(42)
model = NoPadMaskTransformer(
    vocab_size, emb_dim=128, nhead=4, layers=2, num_labels=num_labels, dropout=0.1
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
num_epochs = 5
rec = experiment_data["NoPadMask_Transformer"]["SPR_BENCH"]
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion, None)
    rec["losses"]["train"].append(tr_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["train_macro_f1"].append(tr_f1)
    rec["metrics"]["val_macro_f1"].append(val_f1)
    rec["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | train_F1={tr_f1:.4f} val_F1={val_f1:.4f} (time {time.time()-t0:.1f}s)"
    )

# ------------------------------------------------- test
test_loss, test_f1, test_preds, test_trues = run_epoch(
    model, test_loader, criterion, None
)
rec["test_loss"] = test_loss
rec["test_macro_f1"] = test_f1
rec["predictions"] = test_preds
rec["ground_truth"] = test_trues
print(f"\nTest results: loss={test_loss:.4f} macro_F1={test_f1:.4f}")

# ------------------------------------------------- save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
