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

import os, math, time, json, random
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import pathlib
import matplotlib.pyplot as plt

# ------------------------ working dir / device -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------ helper: load SPR_BENCH ---------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


# detect dataset path
DEFAULT_PATHS = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]
for p in DEFAULT_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found in expected locations.")
print(f"Loading dataset from: {DATA_PATH}")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------------------ metrics ------------------------------------------
def count_color_variety(seq: str):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def entropy_weight(seq: str):
    toks = seq.strip().split()
    if not toks:
        return 0.0
    freqs = Counter(toks)
    total = len(toks)
    ent = -sum((c / total) * math.log2(c / total) for c in freqs.values())
    return ent


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def ewa(seqs, y_true, y_pred):
    w = [entropy_weight(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


# ------------------------ vocab / label mapping ----------------------------
def build_vocab(seqs, min_freq=1):
    cnt = Counter()
    for s in seqs:
        cnt.update(s.strip().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in cnt.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
print(f"Vocab size: {len(vocab)}")

label_set = sorted(set(spr["train"]["label"]))
label2idx = {lbl: i for i, lbl in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
num_labels = len(label2idx)
print(f"Num labels: {num_labels}")


# ------------------------ Torch Dataset ------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, vocab, label2idx):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]
        self.vocab = vocab
        self.label2idx = label2idx

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = [self.vocab.get(t, 1) for t in self.seqs[idx].strip().split()]
        return {
            "input_ids": torch.tensor(toks, dtype=torch.long),
            "length": torch.tensor(len(toks), dtype=torch.long),
            "label": torch.tensor(self.label2idx[self.labels[idx]], dtype=torch.long),
            "seq_raw": self.seqs[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lengths = []
    labels = []
    seq_raw = []
    for i, item in enumerate(batch):
        l = item["length"]
        input_ids[i, :l] = item["input_ids"]
        lengths.append(l)
        labels.append(item["label"])
        seq_raw.append(item["seq_raw"])
    return {
        "input_ids": input_ids,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.stack(labels),
        "seq_raw": seq_raw,
    }


train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ------------------------ Model -------------------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, ids, lengths):
        x = self.emb(ids)  # (B, T, D)
        mask = (ids != 0).unsqueeze(-1)
        x = x * mask
        summed = x.sum(1)
        lengths = lengths.unsqueeze(1).type_as(summed)
        mean = summed / lengths.clamp(min=1)
        return self.fc(mean)


model = MeanEmbedClassifier(len(vocab), 64, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------ experiment_data dict ----------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------ training loop -----------------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    total_loss, n = 0.0, 0
    for batch in train_loader:
        batch_ids = batch["input_ids"].to(device)
        batch_len = batch["lengths"].to(device)
        batch_lab = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(batch_ids, batch_len)
        loss = criterion(logits, batch_lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_ids.size(0)
        n += batch_ids.size(0)
    train_loss = total_loss / n
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))

    # ---- validate ----
    model.eval()
    val_loss, n = 0.0, 0
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch_ids = batch["input_ids"].to(device)
            batch_len = batch["lengths"].to(device)
            batch_lab = batch["labels"].to(device)
            logits = model(batch_ids, batch_len)
            loss = criterion(logits, batch_lab)
            val_loss += loss.item() * batch_ids.size(0)
            n += batch_ids.size(0)
            preds = logits.argmax(dim=1).cpu().tolist()
            labels = batch_lab.cpu().tolist()
            all_seq.extend(batch["seq_raw"])
            all_true.extend([idx2label[i] for i in labels])
            all_pred.extend([idx2label[i] for i in preds])
    val_loss /= n
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))

    cwa_score = cwa(all_seq, all_true, all_pred)
    swa_score = swa(all_seq, all_true, all_pred)
    ewa_score = ewa(all_seq, all_true, all_pred)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (epoch, {"CWA": cwa_score, "SWA": swa_score, "EWA": ewa_score})
    )

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, CWA={cwa_score:.4f}, SWA={swa_score:.4f}, EWA={ewa_score:.4f}"
    )

# ------------------------ test evaluation ---------------------------------
model.eval()
all_seq, all_true, all_pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch_ids = batch["input_ids"].to(device)
        batch_len = batch["lengths"].to(device)
        logits = model(batch_ids, batch_len)
        preds = logits.argmax(dim=1).cpu().tolist()
        labels = batch["labels"].cpu().tolist()
        all_seq.extend(batch["seq_raw"])
        all_true.extend([idx2label[i] for i in labels])
        all_pred.extend([idx2label[i] for i in preds])

experiment_data["SPR_BENCH"]["predictions"] = all_pred
experiment_data["SPR_BENCH"]["ground_truth"] = all_true

test_cwa = cwa(all_seq, all_true, all_pred)
test_swa = swa(all_seq, all_true, all_pred)
test_ewa = ewa(all_seq, all_true, all_pred)
print(f"Test  CWA={test_cwa:.4f}, SWA={test_swa:.4f}, EWA={test_ewa:.4f}")

# ------------------------ save metrics & plot ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot losses
epochs = [e for e, _ in experiment_data["SPR_BENCH"]["losses"]["train"]]
tr_losses = [l for _, l in experiment_data["SPR_BENCH"]["losses"]["train"]]
val_losses = [l for _, l in experiment_data["SPR_BENCH"]["losses"]["val"]]
plt.figure()
plt.plot(epochs, tr_losses, label="train")
plt.plot(epochs, val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
plt.close()
