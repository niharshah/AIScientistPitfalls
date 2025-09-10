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

import os, math, time, pathlib, random, numpy as np
from typing import List, Dict
from collections import Counter

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ----------------------- I/O & PATHS -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------- DATA HELPERS ----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return float(sum(corr)) / float(sum(w)) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return float(sum(corr)) / float(sum(w)) if sum(w) else 0.0


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ----------------------- LOAD DATA ------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", {k: len(v) for k, v in spr.items()})


# ------------------ SYMBOLIC GLYPH CLUSTERING -------------
def token_feature(tok: str) -> List[float]:
    codes = [ord(c) for c in tok]
    first = codes[0]
    rest_mean = sum(codes[1:]) / len(codes[1:]) if len(codes) > 1 else 0.0
    return [first, rest_mean]


all_tokens = sorted(set(t for s in spr["train"]["sequence"] for t in s.split()))
X = np.array([token_feature(t) for t in all_tokens])
k = max(8, int(math.sqrt(len(all_tokens))))
print(f"Clustering {len(all_tokens)} glyphs into {k} clusters …")
glyph2cluster = {
    t: int(c)
    for t, c in zip(
        all_tokens, KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
    )
}
print("Clustering done.")


# ----------------------- DATASET --------------------------
class SPRClusteredDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [glyph2cluster.get(tok, 0) + 1 for tok in self.seqs[idx].split()]
        return {
            "input": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    lens = [len(ex["input"]) for ex in batch]
    max_len = max(lens)
    padded = [
        torch.cat(
            [ex["input"], torch.zeros(max_len - len(ex["input"]), dtype=torch.long)]
        )
        for ex in batch
    ]
    return {
        "input": torch.stack(padded),  # still on CPU
        "len": torch.tensor(lens, dtype=torch.long),
        "label": torch.stack([ex["label"] for ex in batch]),
        "raw_seq": [ex["raw_seq"] for ex in batch],
    }


train_ds, dev_ds, test_ds = (
    SPRClusteredDataset(spr[s]) for s in ["train", "dev", "test"]
)

train_loader = DataLoader(
    train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn
)  # bug fixed
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

num_labels = len(set(spr["train"]["label"]))
vocab_size = k + 2  # +1 for pad, +1 because clusters start at 1
print(f"vocab_size={vocab_size}, num_labels={num_labels}")


# ----------------------- MODEL ----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim, n_class):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_class)

    def forward(self, x, lens):
        x = x.to(device)
        lens = lens.to(device)
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h.squeeze(0))


# -------------------- EXPERIMENT DATA ---------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# --------------------- TRAIN / EVAL -----------------------
def evaluate(model, loader):
    model.eval()
    preds, gts, seqs = [], [], []
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            # move tensors
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["input"], batch_t["len"])
            loss = criterion(logits, batch_t["label"])
            tot_loss += loss.item() * batch_t["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch_t["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    avg_loss = tot_loss / len(loader.dataset)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    return avg_loss, cwa, swa, hwa, preds, gts, seqs


def train_one_lr(lr, epochs=5):
    print(f"\n===== LR={lr:.1e} =====")
    model = GRUClassifier(vocab_size, 32, 64, num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            # move tensors to device
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch_t["input"], batch_t["len"])
            loss = criterion(logits, batch_t["label"])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch_t["label"].size(0)
        tr_loss = epoch_loss / len(train_loader.dataset)

        # validation
        val_loss, cwa, swa, hwa, *_ = evaluate(model, dev_loader)

        experiment_data["SPR_BENCH"]["losses"]["train"].append((lr, ep, tr_loss))
        experiment_data["SPR_BENCH"]["losses"]["val"].append((lr, ep, val_loss))
        experiment_data["SPR_BENCH"]["metrics"]["val"].append((lr, ep, cwa, swa, hwa))

        print(
            f"Epoch {ep}/{epochs} | train_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}"
        )

    # final test evaluation
    test_loss, cwa, swa, hwa, preds, gts, _ = evaluate(model, test_loader)
    print(f"TEST | loss={test_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}")

    experiment_data["SPR_BENCH"]["metrics"]["test"] = (lr, cwa, swa, hwa)
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts
    torch.cuda.empty_cache()


# -------------------- HYPERPARAMETER SWEEP ---------------
for lr in [3e-4, 5e-4, 1e-3, 2e-3]:
    train_one_lr(lr, epochs=5)

# -------------------- SAVE RESULTS -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
