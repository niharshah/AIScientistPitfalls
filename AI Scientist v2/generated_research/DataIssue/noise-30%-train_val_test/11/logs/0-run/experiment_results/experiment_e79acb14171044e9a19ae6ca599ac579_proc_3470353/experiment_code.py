import os, math, pathlib, random, time, json, collections
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
    "hybrid_transformer": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ----------------------------- reproducibility ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ----------------------------- device -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------- data loading -------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict()
    for s, f in [("train", "train.csv"), ("dev", "dev.csv"), ("test", "test.csv")]:
        ds[s] = _load(f)
    return ds


data_path = pathlib.Path(
    os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(data_path)
print({k: len(v) for k, v in spr.items()})

# ----------------------------- vocab --------------------------------------
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
print("vocab_size", vocab_size)


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
print("max_len", max_len)

label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
num_labels = len(label2id)
print("num_labels", num_labels)


# ------------------------ symbolic feature extractor ----------------------
def symbolic_features(seq: str, most_common: List[str]) -> np.ndarray:
    toks = seq.strip().split()
    length = len(toks)
    uniq = len(set(toks))
    ratio = uniq / length if length else 0.0
    repeat = 1.0 if uniq < length else 0.0
    counts = [toks.count(tok) / length for tok in most_common]  # normalized counts
    return np.array([length, uniq, ratio, repeat] + counts, dtype=np.float32)


# determine top 20 tokens
token_freq = collections.Counter(
    t for s in spr["train"]["sequence"] for t in s.strip().split()
)
topK = [t for t, _ in token_freq.most_common(20)]
feat_dim = 4 + len(topK)
print("symbolic feature dim", feat_dim)


# precompute features for speed
def precompute_features(split):
    feats = [symbolic_features(s, topK) for s in split["sequence"]]
    return np.stack(feats)


train_sym = precompute_features(spr["train"])
dev_sym = precompute_features(spr["dev"])
test_sym = precompute_features(spr["test"])


# ----------------------------- Dataset class ------------------------------
class SPRDataset(Dataset):
    def __init__(self, split, sym_feats):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]
        self.sym_feats = sym_feats

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode(self.seqs[idx], max_len), dtype=torch.long
            ),
            "sym_feats": torch.tensor(self.sym_feats[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(
    SPRDataset(spr["train"], train_sym), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    SPRDataset(spr["dev"], dev_sym), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    SPRDataset(spr["test"], test_sym), batch_size=batch_size, shuffle=False
)


# ----------------------------- model --------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridTransformer(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, nhead, nlayer, num_labels, sym_dim, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayer)
        self.sym_mlp = nn.Sequential(
            nn.Linear(sym_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        self.classifier = nn.Linear(emb_dim + 32, num_labels)

    def forward(self, input_ids, sym_feats):
        mask = input_ids == 0
        x = self.embedding(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)  # mean pooling
        sym_vec = self.sym_mlp(sym_feats)
        comb = torch.cat([x, sym_vec], dim=-1)
        return self.classifier(comb)


model = HybridTransformer(vocab_size, 128, 8, 2, num_labels, feat_dim, dropout=0.1).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------- train / eval --------------------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"], batch["sym_feats"])
            loss = criterion(logits, batch["labels"])
            if train:
                loss.backward()
            if train:
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(trues, preds, average="macro")
    return avg_loss, macro_f1, preds, trues


num_epochs = 8
rec = experiment_data["hybrid_transformer"]

for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss, train_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, train=False)

    rec["losses"]["train"].append(train_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["train_macro_f1"].append(train_f1)
    rec["metrics"]["val_macro_f1"].append(val_f1)
    rec["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: training_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
        f"train_F1={train_f1:.4f}, val_F1={val_f1:.4f} (time {time.time()-t0:.1f}s)"
    )

# ----------------------------- final test ----------------------------------
test_loss, test_f1, test_preds, test_trues = run_epoch(test_loader, train=False)
rec["test_loss"] = test_loss
rec["test_macro_f1"] = test_f1
rec["predictions"] = test_preds
rec["ground_truth"] = test_trues
print(f"Test: loss={test_loss:.4f}, macro_F1={test_f1:.4f}")

# ----------------------------- save ----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
