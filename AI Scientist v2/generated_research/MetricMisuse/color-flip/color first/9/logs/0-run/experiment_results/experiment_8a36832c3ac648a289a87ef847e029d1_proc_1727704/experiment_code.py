import os, pathlib, math, random, time
import numpy as np
from collections import Counter
from typing import List, Dict

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ------------------- misc & dirs -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------- helpers -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(
        sum(w), 1e-9
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(
        sum(w), 1e-9
    )


def dwhs(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ------------------- load data ---------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded:", {k: len(v) for k, v in spr.items()})


# ------------- glyph clustering -------------------
def token_feature(tok: str) -> List[float]:
    chars = [ord(c) for c in tok]
    return [chars[0], sum(chars[1:]) / len(chars[1:]) if len(chars) > 1 else 0]


all_tokens = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
X = np.array([token_feature(t) for t in all_tokens])
k = max(8, int(math.sqrt(len(all_tokens))))
print(f"Clustering {len(all_tokens)} glyphs into {k} groups")
km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
glyph2cluster = {tok: int(c) for tok, c in zip(all_tokens, km.labels_)}


# ---------------- dataset --------------------------
class SPRClusteredDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [glyph2cluster.get(t, 0) + 1 for t in self.seqs[idx].split()]
        return {
            "input": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    lens = [len(b["input"]) for b in batch]
    maxlen = max(lens)
    padded = [
        torch.cat([b["input"], torch.zeros(maxlen - len(b["input"]), dtype=torch.long)])
        for b in batch
    ]
    return {
        "input": torch.stack(padded).to(device),
        "label": torch.stack([b["label"] for b in batch]).to(device),
        "len": torch.tensor(lens, dtype=torch.long).to(device),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_ds, dev_ds, test_ds = map(
    SPRClusteredDataset, (spr["train"], spr["dev"], spr["test"])
)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

num_labels = len(set(spr["train"]["label"]))
vocab_size = k + 2  # +PAD+OOV
print("Labels:", num_labels, "vocab:", vocab_size)


# ---------------- model ----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, lens):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h.squeeze(0))


# ------------- hyperparameter tuning ---------------
hidden_sizes = [32, 64, 128, 256]
num_epochs = 5
experiment_data = {"hidden_size": {}}

for hsz in hidden_sizes:
    print(f"\n=== Training with hidden_size={hsz} ===")
    model = GRUClassifier(vocab_size, 32, hsz, num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input"], batch["len"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch["label"].size(0)
        train_loss = epoch_loss / len(train_ds)
        run_data["losses"]["train"].append((epoch, train_loss))

        # ---- dev ----
        model.eval()
        val_loss = 0.0
        all_preds, all_lbls, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["input"], batch["len"])
                loss = criterion(logits, batch["label"])
                val_loss += loss.item() * batch["label"].size(0)
                preds = logits.argmax(-1).cpu().numpy().tolist()
                labels = batch["label"].cpu().numpy().tolist()
                all_preds.extend(preds)
                all_lbls.extend(labels)
                all_seqs.extend(batch["raw_seq"])
        val_loss /= len(dev_ds)
        cwa = color_weighted_accuracy(all_seqs, all_lbls, all_preds)
        swa = shape_weighted_accuracy(all_seqs, all_lbls, all_preds)
        dwh = dwhs(cwa, swa)
        run_data["losses"]["val"].append((epoch, val_loss))
        run_data["metrics"]["val"].append((epoch, cwa, swa, dwh))
        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} DWHS={dwh:.3f}"
        )

    # ---- test ----
    model.eval()
    t_preds, t_lbls, t_seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input"], batch["len"])
            preds = logits.argmax(-1).cpu().numpy().tolist()
            labels = batch["label"].cpu().numpy().tolist()
            t_preds.extend(preds)
            t_lbls.extend(labels)
            t_seqs.extend(batch["raw_seq"])
    cwa = color_weighted_accuracy(t_seqs, t_lbls, t_preds)
    swa = shape_weighted_accuracy(t_seqs, t_lbls, t_preds)
    dwh = dwhs(cwa, swa)
    print(f"TEST hidden={hsz} | CWA={cwa:.3f} SWA={swa:.3f} DWHS={dwh:.3f}")
    run_data["metrics"]["test"] = (cwa, swa, dwh)
    run_data["predictions"] = t_preds
    run_data["ground_truth"] = t_lbls

    experiment_data["hidden_size"][hsz] = {"SPR_BENCH": run_data}

    # free memory
    del model
    torch.cuda.empty_cache()

# ------------- save --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
