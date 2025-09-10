import os, math, random, time, json, pathlib
from collections import Counter
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH_symbolic": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- reproducibility ----------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ---------------- data loading -------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(file_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / file_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------- vocabulary & helper ------------------------------------
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
print("vocab_size:", vocab_size)

# determine max_len for padding
max_len = min(64, max(len(s.split()) for s in spr["train"]["sequence"]))
print("max_len:", max_len)

# label mapping
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
num_labels = len(label2id)
print("num_labels:", num_labels)

# ---------------- symbolic feature design ---------------------------------
# pick top-K tokens to track presence/count explicitly
TOP_K = 20
tok_counter = Counter()
for seq in spr["train"]["sequence"]:
    tok_counter.update(seq.strip().split())
top_k_tokens = [tok for tok, _ in tok_counter.most_common(TOP_K)]
tok2kidx = {tok: i for i, tok in enumerate(top_k_tokens)}


def compute_symbolic_features(tokens: List[str]) -> np.ndarray:
    seq_len = len(tokens)
    uniq_cnt = len(set(tokens))
    features = np.zeros(4 + TOP_K, dtype=np.float32)
    # scalar features
    features[0] = seq_len / max_len  # normalized length
    features[1] = uniq_cnt / max_len  # normalized unique count
    tok_ids = [vocab.get(t, 1) for t in tokens]
    features[2] = np.mean(tok_ids) / vocab_size  # normalized mean id
    features[3] = (np.std(tok_ids) if seq_len > 0 else 0) / vocab_size
    # top-K token presence/count (normalized by length)
    for t in tokens:
        if t in tok2kidx:
            features[4 + tok2kidx[t]] += 1.0
    if seq_len > 0:
        features[4:] /= seq_len
    return features


SYM_DIM = 64
NUM_SYM_FEATS = 4 + TOP_K


def encode_tokens(tokens: List[str]) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokens][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


# ---------------- dataset -------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = self.seqs[idx].strip().split()
        return {
            "input_ids": torch.tensor(encode_tokens(tokens), dtype=torch.long),
            "sym_feats": torch.tensor(
                compute_symbolic_features(tokens), dtype=torch.float32
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 128
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# ---------------- model ---------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SymbolicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        nhead,
        nlayer,
        num_labels,
        num_sym_feats,
        sym_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayer)
        self.sym_mlp = nn.Sequential(
            nn.Linear(num_sym_feats, sym_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(emb_dim + sym_dim, num_labels)

    def forward(self, input_ids, sym_feats):
        mask = input_ids == 0
        x = self.embedding(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)  # mean pooling
        s = self.sym_mlp(sym_feats)
        fused = torch.cat([x, s], dim=-1)
        return self.classifier(fused)


# ---------------- helpers -------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    total_loss, all_preds, all_trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad() if train_flag else None
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"], batch["sym_feats"])
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


# ---------------- training ------------------------------------------------
dropout = 0.1
model = SymbolicTransformer(
    vocab_size, 128, 8, 2, num_labels, NUM_SYM_FEATS, SYM_DIM, dropout=dropout
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 12
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)
    rec = experiment_data["SPR_BENCH_symbolic"]
    rec["losses"]["train"].append(tr_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["train_macro_f1"].append(tr_f1)
    rec["metrics"]["val_macro_f1"].append(val_f1)
    rec["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}  "
        f"(train_loss={tr_loss:.4f}, time {time.time()-t0:.1f}s)"
    )

# ---------------- final evaluation ---------------------------------------
test_loss, test_f1, test_preds, test_trues = run_epoch(model, test_loader, criterion)
print(f"\nTest: loss={test_loss:.4f} macro_f1={test_f1:.4f}")
rec["test_loss"] = test_loss
rec["test_macro_f1"] = test_f1
rec["predictions"] = test_preds
rec["ground_truth"] = test_trues

# ---------------- save ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
