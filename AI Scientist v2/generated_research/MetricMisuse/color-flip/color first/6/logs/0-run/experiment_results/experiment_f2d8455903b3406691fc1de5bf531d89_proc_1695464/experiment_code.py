import os, pathlib, random, time, math
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- util to load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path):
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


# try typical locations -------------------------------------------------
for _p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]:
    if _p.exists():
        DATA_PATH = _p
        break
else:
    raise FileNotFoundError("Place SPR_BENCH folder next to this script.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- clustering: glyph -> shape id ----------
def cluster_token(tok: str) -> str:
    return tok[0] if tok else "<pad>"


shapes = sorted(
    {cluster_token(t) for s in spr["train"]["sequence"] for t in s.strip().split()}
)
shape2idx = {s: i + 1 for i, s in enumerate(shapes)}  # 0 kept for pad
vocab_size = len(shape2idx) + 1
print(f"Shape-clusters: {len(shape2idx)}")

labels = sorted(set(spr["train"]["label"]))
lab2idx = {l: i for i, l in enumerate(labels)}
idx2lab = {i: l for l, i in lab2idx.items()}


# ---------- metrics ----------
def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def pcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- dataset ----------
class ClusteredSPR(Dataset):
    def __init__(self, hf, shape2idx, lab2idx):
        self.seqs = hf["sequence"]
        self.labs = hf["label"]
        self.s2i = shape2idx
        self.l2i = lab2idx

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        clusters = [self.s2i.get(cluster_token(t), 0) for t in self.seqs[idx].split()]
        return {
            "ids": torch.tensor(clusters, dtype=torch.long),
            "len": len(clusters),
            "label": self.l2i[self.labs[idx]],
            "raw": self.seqs[idx],
        }


def collate_fn(batch):
    max_len = max(b["len"] for b in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    lens = []
    labels = []
    raws = []
    for i, b in enumerate(batch):
        ids[i, : b["len"]] = b["ids"]
        lens.append(b["len"])
        labels.append(b["label"])
        raws.append(b["raw"])
    return {
        "ids": ids,
        "lens": torch.tensor(lens),
        "labels": torch.tensor(labels),
        "raw": raws,
    }


train_ds = ClusteredSPR(spr["train"], shape2idx, lab2idx)
dev_ds = ClusteredSPR(spr["dev"], shape2idx, lab2idx)
test_ds = ClusteredSPR(spr["test"], shape2idx, lab2idx)

train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)


# ---------- model ----------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vsz, edim, hdim, n_lbl):
        super().__init__()
        self.emb = nn.Embedding(vsz, edim, padding_idx=0)
        self.lstm = nn.LSTM(edim, hdim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hdim, n_lbl)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # use mean over time steps
        mask = (ids != 0).unsqueeze(-1)
        mean = (out * mask).sum(1) / lens.unsqueeze(1).type_as(out)
        return self.fc(mean)


# ---------- experiment ----------
EPOCHS = 5
BATCH = 128
beta2 = 0.98
experiment_data = {
    "cluster_shape": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

model = BiLSTMClassifier(vocab_size, 64, 128, len(lab2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, beta2))


def evaluate(loader):
    model.eval()
    seqs, true, pred = [], [], []
    tot_loss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            lens = batch["lens"].to(device)
            labs = batch["labels"].to(device)
            logits = model(ids, lens)
            loss = criterion(logits, labs)
            tot_loss += loss.item() * ids.size(0)
            n += ids.size(0)
            pr = logits.argmax(1).cpu().tolist()
            tr = labs.cpu().tolist()
            seqs.extend(batch["raw"])
            true.extend([idx2lab[i] for i in tr])
            pred.extend([idx2lab[i] for i in pr])
    avg_loss = tot_loss / n
    return avg_loss, seqs, true, pred


for epoch in range(1, EPOCHS + 1):
    model.train()
    tot, n = 0, 0
    for batch in train_loader(BATCH):
        ids = batch["ids"].to(device)
        lens = batch["lens"].to(device)
        labs = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, lens), labs)
        loss.backward()
        optimizer.step()
        tot += loss.item() * ids.size(0)
        n += ids.size(0)
    tr_loss = tot / n
    experiment_data["cluster_shape"]["losses"]["train"].append((epoch, tr_loss))

    val_loss, seqs, y_true, y_pred = evaluate(dev_loader)
    experiment_data["cluster_shape"]["losses"]["val"].append((epoch, val_loss))
    cwa_v, swa_v, pcwa_v = (
        cwa(seqs, y_true, y_pred),
        swa(seqs, y_true, y_pred),
        pcwa(seqs, y_true, y_pred),
    )
    experiment_data["cluster_shape"]["metrics"]["val"].append(
        (epoch, {"CWA": cwa_v, "SWA": swa_v, "PCWA": pcwa_v})
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"CWA {cwa_v:.4f} | SWA {swa_v:.4f} | PCWA {pcwa_v:.4f}"
    )

# ---------- test ----------
_, seqs, y_true, y_pred = evaluate(test_loader)
experiment_data["cluster_shape"]["predictions"] = y_pred
experiment_data["cluster_shape"]["ground_truth"] = y_true
tcwa, tswa, tpcwa = (
    cwa(seqs, y_true, y_pred),
    swa(seqs, y_true, y_pred),
    pcwa(seqs, y_true, y_pred),
)
print(f"Test  CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
