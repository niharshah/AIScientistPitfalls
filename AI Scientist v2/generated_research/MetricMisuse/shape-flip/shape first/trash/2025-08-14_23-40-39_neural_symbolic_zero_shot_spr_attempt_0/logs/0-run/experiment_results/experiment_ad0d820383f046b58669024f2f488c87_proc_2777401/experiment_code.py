# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, math, pathlib, numpy as np, torch
from collections import Counter
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ----------------- Paths / working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- Disable HF cache -----------------
disable_caching()

# ----------------- Experiment data container -----------------
experiment_data = {
    "epochs_tuning": {
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
}

# ----------------- Hyper-parameters -----------------
EMB_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 20  # extended training
LR = 1e-3
PATIENCE = 3  # early-stopping patience (epochs w/o BPS improvement)
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"


# ----------------- Dataset locating helpers -----------------
def resolve_spr_path() -> pathlib.Path:
    env = os.getenv("SPR_PATH")
    if env and (pathlib.Path(env) / "train.csv").exists():
        print("[Data] Using SPR_BENCH from SPR_PATH=", env)
        return pathlib.Path(env)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            print("[Data] Found SPR_BENCH at", p / "SPR_BENCH")
            return p / "SPR_BENCH"
    fb = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fb / "train.csv").exists():
        print("[Data] Using fallback path", fb)
        return fb
    raise FileNotFoundError("SPR_BENCH not found; set SPR_PATH.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ----------------- Load data & build vocab -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)

train_sequences = spr["train"]["sequence"]
token_counter = Counter(tok for seq in train_sequences for tok in seq.split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in token_counter:
    vocab[tok] = len(vocab)
inv_vocab = {i: t for t, i in vocab.items()}

label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)
print(f"Vocab size {len(vocab)} | Classes {NUM_CLASSES}")


def encode_sequence(seq: str):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return dict(
            input_ids=torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long),
            labels=torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
            seq_str=self.seqs[idx],
        )


def collate_fn(batch):
    lens = [len(b["input_ids"]) for b in batch]
    max_len = max(lens)
    inputs = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, b in enumerate(batch):
        inputs[i, : lens[i]] = b["input_ids"]
    return dict(
        input_ids=inputs,
        labels=torch.stack([b["labels"] for b in batch]),
        seq_strs=[b["seq_str"] for b in batch],
    )


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
    def __init__(self, vocab_sz, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc1, self.relu = nn.Linear(emb_dim, HIDDEN_DIM), nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        avg = (self.emb(x) * mask).sum(1) / (mask.sum(1).clamp(min=1e-6))
        return self.fc2(self.relu(self.fc1(avg)))


model = SPRClassifier(len(vocab), EMB_DIM, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)


# ----------------- Evaluation helper -----------------
def evaluate(loader):
    model.eval()
    tot_loss = 0
    n = 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            p = logits.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            preds.extend(p)
            labels.extend(l)
            seqs.extend(batch["seq_strs"])
    loss = tot_loss / max(n, 1)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    bps = math.sqrt(swa * cwa) if swa >= 0 and cwa >= 0 else 0.0
    return loss, swa, cwa, bps, preds, labels


# ----------------- Training w/ early stopping -----------------
best_bps = -1
patience_ctr = 0
best_state = None
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss = 0
    seen = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optim.zero_grad()
        loss = criterion(model(batch["input_ids"]), batch["labels"])
        loss.backward()
        optim.step()
        run_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)
    train_loss = run_loss / seen

    val_loss, swa, cwa, bps, _, _ = evaluate(dev_loader)
    print(
        f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | SWA {swa:.4f} | CWA {cwa:.4f} | BPS {bps:.4f}"
    )

    # log
    m = experiment_data["epochs_tuning"]["SPR_BENCH"]["metrics"]
    m["train_loss"].append(train_loss)
    m["val_loss"].append(val_loss)
    m["val_swa"].append(swa)
    m["val_cwa"].append(cwa)
    m["val_bps"].append(bps)
    experiment_data["epochs_tuning"]["SPR_BENCH"]["timestamps"].append(
        datetime.utcnow().isoformat()
    )

    # early stopping check
    if bps > best_bps:
        best_bps = bps
        best_state = model.state_dict()
        patience_ctr = 0
    else:
        patience_ctr += 1
    if patience_ctr >= PATIENCE:
        print("Early stopping triggered.")
        break

# ----------------- Restore best model -----------------
if best_state is not None:
    model.load_state_dict(best_state)

# ----------------- Final evaluation -----------------
dev_loss, dev_swa, dev_cwa, dev_bps, dev_preds, dev_labels = evaluate(dev_loader)
test_loss, test_swa, test_cwa, test_bps, test_preds, test_labels = evaluate(test_loader)

print(
    f"=== DEV  === loss {dev_loss:.4f} | SWA {dev_swa:.4f} | CWA {dev_cwa:.4f} | BPS {dev_bps:.4f}"
)
print(
    f"=== TEST === loss {test_loss:.4f} | SWA {test_swa:.4f} | CWA {test_cwa:.4f} | BPS {test_bps:.4f}"
)

d = experiment_data["epochs_tuning"]["SPR_BENCH"]
d["predictions"]["dev"] = dev_preds
d["ground_truth"]["dev"] = dev_labels
d["predictions"]["test"] = test_preds
d["ground_truth"]["test"] = test_labels

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
