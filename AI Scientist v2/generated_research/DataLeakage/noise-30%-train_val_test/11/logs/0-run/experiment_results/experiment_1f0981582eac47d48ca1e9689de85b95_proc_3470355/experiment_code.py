import os, math, pathlib, random, time, json
from collections import Counter
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ------------------- working dir & bookkeeping ----------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "baseline": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    },
    "neuro_symbolic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    },
}


# ------------------- reproducibility --------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ------------------- device -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- data loading -----------------------------------------------------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp, fname in [("train", "train.csv"), ("dev", "dev.csv"), ("test", "test.csv")]:
        d[sp] = _load(fname)
    return d


# try to locate dataset automatically
candidates = [
    os.getenv("SPR_PATH"),
    "SPR_BENCH",
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
]
data_root = None
for cand in candidates:
    if cand and pathlib.Path(cand).exists():
        data_root = pathlib.Path(cand)
        break
if data_root is None:
    raise FileNotFoundError(
        "SPR_BENCH dataset folder not found. "
        "Set env var SPR_PATH or place folder in cwd."
    )
print("Using dataset folder:", data_root)
spr = load_spr_bench(data_root)
print({k: len(v) for k, v in spr.items()})

# ------------------- vocab + encoding -------------------------------------------------
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
print("Vocab size:", vocab_size)


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = min(64, max(len(s.split()) for s in spr["train"]["sequence"]))
print("Max length set to", max_len)

label_set = sorted(list(set(spr["train"]["label"])))
label2id = {lab: i for i, lab in enumerate(label_set)}
num_labels = len(label2id)
print("Num labels:", num_labels)


# ------------------- symbolic feature extraction --------------------------------------
def sym_feats(tokens: List[str]) -> List[float]:
    seq_len = len(tokens)
    uniq = len(set(tokens))
    uniq_ratio = uniq / seq_len
    repeat = seq_len - uniq
    max_freq = max(Counter(tokens).values())
    return [seq_len, uniq, uniq_ratio, repeat, max_freq]


# collect train statistics for min-max scaling
all_feats = [sym_feats(s.split()) for s in spr["train"]["sequence"]]
all_feats = np.array(all_feats)
feat_min = all_feats.min(axis=0)
feat_max = all_feats.max(axis=0)


def norm_feats(feat_vec: List[float]) -> List[float]:
    v = np.array(feat_vec)
    return ((v - feat_min) / (feat_max - feat_min + 1e-8)).tolist()


# ------------------- Dataset obj ------------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]
        self.sym = [norm_feats(sym_feats(s.split())) for s in self.seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode(self.seqs[idx], max_len), dtype=torch.long
            ),
            "sym_feats": torch.tensor(self.sym[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# ------------------- model defs -------------------------------------------------------
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


class TransformerBaseline(nn.Module):
    def __init__(
        self, vocab_size, emb_dim=128, nhead=8, num_layers=2, num_labels=10, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(emb_dim, num_labels)

    def forward(self, input_ids, sym_feats=None):
        pad_mask = input_ids == 0
        x = self.embedding(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0).mean(1)
        return self.cls(x)


class NeuroSymbolic(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        nhead=8,
        num_layers=2,
        num_labels=10,
        sym_dim=5,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.sym_mlp = nn.Sequential(
            nn.Linear(sym_dim, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )
        self.cls = nn.Linear(emb_dim + 64, num_labels)

    def forward(self, input_ids, sym_feats):
        pad_mask = input_ids == 0
        x = self.embedding(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0).mean(1)  # [B, emb_dim]
        sym_vec = self.sym_mlp(sym_feats)  # [B, 64]
        cat = torch.cat([x, sym_vec], dim=-1)
        return self.cls(cat)


# ------------------- training utilities ----------------------------------------------
def run_epoch(model, loader, crit, optim=None):
    train_flag = optim is not None
    model.train() if train_flag else model.eval()
    total_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if train_flag:
            optim.zero_grad()
        with torch.set_grad_enabled(train_flag):
            # Some models don't use sym_feats (baseline)
            logits = model(batch["input_ids"], batch.get("sym_feats"))
            loss = crit(logits, batch["labels"])
            if train_flag:
                loss.backward()
                optim.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).detach().cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(trues, preds, average="macro")
    return avg_loss, macro_f1, preds, trues


# ------------------- experiment loop --------------------------------------------------
def train_and_eval(model_name: str, model, num_epochs: int = 5):
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, crit, optim)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, crit, None)
        ed = experiment_data[model_name]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(tr_f1)
        ed["metrics"]["val"].append(val_f1)
        ed["epochs"].append(epoch)
        print(
            f"[{model_name}] Epoch {epoch}: "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f} "
            f"({time.time()-t0:.1f}s)"
        )
    # test evaluation
    test_loss, test_f1, preds, trues = run_epoch(model, test_loader, crit, None)
    ed["test_loss"] = test_loss
    ed["test_macro_f1"] = test_f1
    ed["predictions"] = preds
    ed["ground_truth"] = trues
    print(f"[{model_name}] TEST: loss={test_loss:.4f} macro_F1={test_f1:.4f}")


# ------------------- run both models --------------------------------------------------
baseline_model = TransformerBaseline(vocab_size, num_labels=num_labels)
ns_model = NeuroSymbolic(vocab_size, num_labels=num_labels)

train_and_eval("baseline", baseline_model)
train_and_eval("neuro_symbolic", ns_model)

# ------------------- save everything --------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
