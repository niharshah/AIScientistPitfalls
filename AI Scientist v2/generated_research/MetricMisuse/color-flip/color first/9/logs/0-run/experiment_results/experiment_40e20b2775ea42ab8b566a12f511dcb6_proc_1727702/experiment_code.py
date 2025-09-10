# embedding_dim hyper-parameter sweep for SPR-BENCH
import os, pathlib, random, math, time
import numpy as np
from collections import Counter
from typing import List, Dict

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# ---- reproducibility --------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---- device -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- paths / experiment dict -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "embedding_dim_tuning": {  # <-- hyper-param tuning type
        "SPR_BENCH": {}  # placeholders per emb_dim will be filled later
    }
}


# ---- helpers for metrics ----------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return float(sum(c)) / float(sum(w)) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return float(sum(c)) / float(sum(w)) if sum(w) else 0.0


def dwhs(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ---- load SPR-BENCH ---------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # local helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded:", {k: len(v) for k, v in spr.items()})


# ---- glyph clustering --------------------------------------------------------
def token_feature(tok: str) -> List[float]:
    codepoints = [ord(c) for c in tok]
    first = codepoints[0]
    rest_mean = (
        sum(codepoints[1:]) / len(codepoints[1:]) if len(codepoints) > 1 else 0.0
    )
    return [first, rest_mean]


all_tokens = sorted({t for seq in spr["train"]["sequence"] for t in seq.split()})
X = np.array([token_feature(t) for t in all_tokens])
k = max(8, int(math.sqrt(len(all_tokens))))
print(f"Clustering {len(all_tokens)} tokens into {k} clusters …")
clusters = KMeans(n_clusters=k, random_state=SEED, n_init="auto").fit_predict(X)
glyph2cluster = {tok: int(c) for tok, c in zip(all_tokens, clusters)}
print("Clustering done.")


# ---- dataset / dataloader ----------------------------------------------------
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
        torch.cat([b["input"], torch.zeros(maxlen - l, dtype=torch.long)])
        for b, l in zip(batch, lens)
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
vocab_size = k + 2  # clusters + PAD(0) + OOV
print(f"num_labels={num_labels}, vocab_size={vocab_size}")


# ---- model definition --------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, lens):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h.squeeze(0))


# ---- hyper-parameter sweep ---------------------------------------------------
embed_dims = [16, 32, 64, 128]
num_epochs = 5

for emb_dim in embed_dims:
    tag = f"emb_{emb_dim}"
    print(f"\n=== Training with embedding_dim={emb_dim} ===")
    # prepare storage in experiment_data
    experiment_data["embedding_dim_tuning"]["SPR_BENCH"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = GRUClassifier(vocab_size, emb_dim, 64, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------- training loop ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input"], batch["len"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            running += loss.item() * batch["label"].size(0)
        tr_loss = running / len(train_ds)
        experiment_data["embedding_dim_tuning"]["SPR_BENCH"][tag]["losses"][
            "train"
        ].append((epoch, tr_loss))

        # -------- validation ----------
        model.eval()
        v_loss = 0.0
        pr, lb, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["input"], batch["len"])
                loss = criterion(logits, batch["label"])
                v_loss += loss.item() * batch["label"].size(0)
                preds = logits.argmax(-1).cpu().tolist()
                pr.extend(preds)
                lb.extend(batch["label"].cpu().tolist())
                seqs.extend(batch["raw_seq"])
        v_loss /= len(dev_ds)
        cwa = color_weighted_accuracy(seqs, lb, pr)
        swa = shape_weighted_accuracy(seqs, lb, pr)
        v_dwhs = dwhs(cwa, swa)
        experiment_data["embedding_dim_tuning"]["SPR_BENCH"][tag]["losses"][
            "val"
        ].append((epoch, v_loss))
        experiment_data["embedding_dim_tuning"]["SPR_BENCH"][tag]["metrics"][
            "val"
        ].append((epoch, cwa, swa, v_dwhs))
        print(
            f"Epoch {epoch}: val_loss={v_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} DWHS={v_dwhs:.3f}"
        )

    # -------- test evaluation ----------
    model.eval()
    pr, lb, seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input"], batch["len"])
            preds = logits.argmax(-1).cpu().tolist()
            pr.extend(preds)
            lb.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    cwa = color_weighted_accuracy(seqs, lb, pr)
    swa = shape_weighted_accuracy(seqs, lb, pr)
    test_dwhs = dwhs(cwa, swa)
    print(
        f"TEST (emb_dim={emb_dim}) | CWA={cwa:.3f} SWA={swa:.3f} DWHS={test_dwhs:.3f}"
    )
    edict = experiment_data["embedding_dim_tuning"]["SPR_BENCH"][tag]
    edict["predictions"], edict["ground_truth"] = pr, lb
    edict["metrics"]["test"] = (cwa, swa, test_dwhs)

    # free GPU mem between runs
    del model, optimizer, criterion
    torch.cuda.empty_cache()

# ---- save -------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
