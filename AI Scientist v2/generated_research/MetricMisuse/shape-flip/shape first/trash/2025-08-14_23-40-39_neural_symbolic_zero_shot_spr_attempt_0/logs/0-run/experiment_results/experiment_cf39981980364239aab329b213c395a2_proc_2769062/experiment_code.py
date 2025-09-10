import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import math
import pathlib
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Disable HF global cache to avoid clutter -------------
disable_caching()


# ----------------- Data-set path resolver -----------------
def resolve_spr_path() -> pathlib.Path:
    """
    1. Use env var SPR_PATH if it exists and looks valid
    2. Walk up parent directories from cwd looking for SPR_BENCH/
    3. Fallback to historical absolute path shipped with repo
    """
    # 1) environment variable
    env_path = os.getenv("SPR_PATH")
    if env_path:
        p = pathlib.Path(env_path).expanduser()
        if (p / "train.csv").exists():
            print(f"[Data] Using SPR_BENCH from SPR_PATH={p}")
            return p

    # 2) parent-dir walk
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        candidate = parent / "SPR_BENCH"
        if (candidate / "train.csv").exists():
            print(f"[Data] Found SPR_BENCH at {candidate}")
            return candidate

    # 3) fallback absolute path (the one seen in bug report)
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        print(f"[Data] Using fallback SPR_BENCH at {fallback}")
        return fallback

    raise FileNotFoundError(
        "Cannot locate SPR_BENCH. Please set SPR_PATH env variable "
        "or place the dataset in a parent directory."
    )


# ----------------- SPR utilities (copied) -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",  # treat csv as a single split
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# ----------------- Experiment data container -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "val_swa": [],
            "val_cwa": [],
            "val_bps": [],
        },
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
        "timestamps": [],
    }
}

# ----------------- Hyper-parameters -----------------
EMB_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# ----------------- Dataset & Vocabulary build -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)

train_sequences = spr["train"]["sequence"]
token_counter = Counter(tok for seq in train_sequences for tok in seq.strip().split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in token_counter:
    vocab[tok] = len(vocab)
inv_vocab = {i: t for t, i in vocab.items()}

label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)
print(f"Vocab size: {len(vocab)} | Classes: {NUM_CLASSES}")


def encode_sequence(seq: str):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.strip().split()]


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode_sequence(self.seqs[idx]), dtype=torch.long
            ),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
            "seq_str": self.seqs[idx],
        }


def collate_fn(batch):
    lengths = [len(item["input_ids"]) for item in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
    labels = torch.stack([item["labels"] for item in batch])
    seq_strs = [item["seq_str"] for item in batch]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "seq_strs": seq_strs,
        "lengths": torch.tensor(lengths),
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ----------------- Model -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, input_ids):
        mask = (input_ids != 0).float().unsqueeze(-1)  # (B,L,1)
        emb = self.emb(input_ids)  # (B,L,E)
        summed = (emb * mask).sum(1)  # (B,E)
        lengths = mask.sum(1).clamp(min=1e-6)
        avg = summed / lengths
        x = self.relu(self.fc1(avg))
        logits = self.fc2(x)
        return logits


model = SPRClassifier(len(vocab), EMB_DIM, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ----------------- Evaluation helper -----------------
def evaluate(loader):
    model.eval()
    tot_loss, n_items = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n_items += bs
            preds = logits.argmax(1).cpu().numpy().tolist()
            labels = batch["labels"].cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(batch["seq_strs"])
    avg_loss = tot_loss / max(n_items, 1)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    bps = math.sqrt(swa * cwa) if swa >= 0 and cwa >= 0 else 0.0
    return avg_loss, swa, cwa, bps, all_preds, all_labels, all_seqs


# ----------------- Training loop -----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, seen = 0.0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)
    train_loss = running_loss / seen

    val_loss, swa, cwa, bps, *_ = evaluate(dev_loader)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | BPS={bps:.4f}"
    )

    # store metrics
    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_cwa"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["val_bps"].append(bps)
    experiment_data["SPR_BENCH"]["timestamps"].append(datetime.utcnow().isoformat())

# ----------------- Final evaluation -----------------
dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels, _ = evaluate(dev_loader)
test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels, _ = evaluate(
    test_loader
)

print(
    f"=== Final DEV ===  loss {dev_loss:.4f} | SWA {dev_swa:.4f} | "
    f"CWA {dev_cwa:.4f} | BPS {dev_bps:.4f}"
)
print(
    f"=== Final TEST === loss {test_loss:.4f} | SWA {test_swa:.4f} | "
    f"CWA {test_cwa:.4f} | BPS {test_bps:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_preds
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_labels
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_labels

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
