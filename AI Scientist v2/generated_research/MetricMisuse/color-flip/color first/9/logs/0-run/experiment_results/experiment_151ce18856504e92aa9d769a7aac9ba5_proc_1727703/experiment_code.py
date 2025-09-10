import os, pathlib, random, math, time, numpy as np
from typing import List, Dict
from collections import Counter

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ----------------------------------------------------
# basic working dir & device
# ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------
# helpers copied from baseline
# ----------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for split in ["train", "dev", "test"]:
        dd[split] = _load(f"{split}.csv")
    return dd


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


def harmonic_weighted_accuracy(cwa, swa):  # aka DWHS in baseline
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ----------------------------------------------------
# load dataset
# ----------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", {k: len(v) for k, v in spr.items()})


# ----------------------------------------------------
# glyph clustering
# ----------------------------------------------------
def token_feature(tok: str) -> List[float]:
    codes = [ord(c) for c in tok]
    first = codes[0]
    rest_mean = sum(codes[1:]) / len(codes[1:]) if len(codes) > 1 else 0.0
    return [first, rest_mean]


all_tokens = sorted({tok for s in spr["train"]["sequence"] for tok in s.split()})
X = np.array([token_feature(t) for t in all_tokens])
k = max(8, int(math.sqrt(len(all_tokens))))
print(f"Clustering {len(all_tokens)} glyphs into {k} clusters …")
glyph2cluster = {
    t: int(c)
    for t, c in zip(
        all_tokens, KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
    )
}
print("Cluster mapping built.")


# ----------------------------------------------------
# torch Dataset / DataLoader
# ----------------------------------------------------
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
    lens = [len(x["input"]) for x in batch]
    maxlen = max(lens)
    padded = [
        torch.cat([x["input"], torch.zeros(maxlen - len(x["input"]), dtype=torch.long)])
        for x in batch
    ]
    return {
        "input": torch.stack(padded),  # to be moved to device later
        "label": torch.stack([x["label"] for x in batch]),
        "len": torch.tensor(lens, dtype=torch.long),
        "raw_seq": [x["raw_seq"] for x in batch],
    }


train_ds, dev_ds, test_ds = (
    SPRClusteredDataset(spr[s]) for s in ["train", "dev", "test"]
)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

# ----------------------------------------------------
# model
# ----------------------------------------------------
num_labels = len(set(spr["train"]["label"]))
vocab_size = k + 2  # clusters + padding_idx(0)
print(f"num_labels={num_labels}, vocab_size={vocab_size}")


class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb, hidden, n_class):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x, lens):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h.squeeze(0))


# ----------------------------------------------------
# experiment tracking structure
# ----------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ----------------------------------------------------
# training routine
# ----------------------------------------------------
def run_experiment(lr: float, epochs: int = 5):
    model = GRUClassifier(vocab_size, 32, 64, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        # ------------- training -------------
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input"], batch["len"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["label"].size(0)
        tr_loss = tot_loss / len(train_ds)
        experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))

        # ------------- validation -----------
        model.eval()
        tot_loss = 0.0
        preds, gts, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
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
        val_loss = tot_loss / len(dev_ds)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hwa = harmonic_weighted_accuracy(cwa, swa)
        experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, val_loss))
        experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, cwa, swa, hwa))
        print(
            f"Epoch {ep}: validation_loss = {val_loss:.4f} "
            f"CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}"
        )

    # ------------- test -------------------
    model.eval()
    preds, gts, seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["input"], batch_t["len"])
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch_t["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    cwa = color_weighted_accuracy(seqs, gts, preds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(cwa, swa)
    print(f"TEST RESULTS — CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}")

    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts
    experiment_data["SPR_BENCH"]["metrics"]["test"] = (cwa, swa, hwa)
    torch.cuda.empty_cache()


# ----------------------------------------------------
# hyper-parameter sweep (learning rate)
# ----------------------------------------------------
for lr in [3e-4, 5e-4, 1e-3, 2e-3]:
    print(f"\n=== Running experiment with lr={lr} ===")
    run_experiment(lr, epochs=3)  # fewer epochs for quick sweep

# ----------------------------------------------------
# save all results
# ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
