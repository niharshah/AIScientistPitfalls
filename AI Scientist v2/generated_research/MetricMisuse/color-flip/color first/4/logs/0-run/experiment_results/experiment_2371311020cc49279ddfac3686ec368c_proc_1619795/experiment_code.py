import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

# ---------------- GPU set-up ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------- Dataset location -----------
def find_spr_bench(start_dir: pathlib.Path = pathlib.Path.cwd()) -> pathlib.Path:
    """Return a pathlib.Path pointing to SPR_BENCH that has train/dev/test csv files."""
    # 1) environment variable takes precedence
    env_dir = os.getenv("SPR_DATA_DIR")
    if env_dir:
        p = pathlib.Path(env_dir).expanduser().resolve()
        if (p / "train.csv").exists():
            return p
    # 2) search current dir and all parents
    for path in [start_dir] + list(start_dir.parents):
        cand = path / "SPR_BENCH"
        if (
            (cand / "train.csv").exists()
            and (cand / "dev.csv").exists()
            and (cand / "test.csv").exists()
        ):
            return cand
    # 3) not found
    raise FileNotFoundError(
        "Could not locate SPR_BENCH dataset. "
        "Place it somewhere above the current directory or set SPR_DATA_DIR."
    )


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# -------------- SPR utilities --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def pattern_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------- Load dataset ---------------
spr = load_spr_bench(DATA_PATH)


# -------------- Build vocab ----------------
def tokenize(seq):
    return seq.strip().split()


vocab = {"<pad>": 0}
for ex in spr["train"]:
    for tok in tokenize(ex["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")


# ------------- Dataset wrappers -----------
def encode_sequence(seq):
    return torch.tensor([vocab[tok] for tok in tokenize(seq)], dtype=torch.long)


class SPRSet(torch.utils.data.Dataset):
    def __init__(self, hf_split):
        self.seqs = [ex["sequence"] for ex in hf_split]
        self.labels = torch.tensor([ex["label"] for ex in hf_split], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": encode_sequence(self.seqs[idx]),
            "label": self.labels[idx],
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    seqs = [b["seq"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return {"seq": padded, "label": labels, "raw_seq": raw}


train_loader = DataLoader(
    SPRSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)

num_labels = int(max(spr["train"]["label"])) + 1
print(f"Num labels: {num_labels}")


# -------------- Model ----------------------
class AvgEmbClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)  # (B,T,1)
        summed = (self.emb(x) * mask).sum(1)  # (B,E)
        lens = mask.sum(1).clamp(min=1)
        avg = summed / lens
        return self.fc(avg)


model = AvgEmbClassifier(vocab_size, 64, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------- Experiment bookkeeping -----
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------- Training loop --------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    # ---- training ----
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["seq"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["label"].size(0)
    train_loss = running_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))

    # ---- validation ----
    model.eval()
    val_running_loss = 0.0
    all_preds, all_labels, all_raw = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["seq"])
            loss = criterion(logits, batch["label"])
            val_running_loss += loss.item() * batch["label"].size(0)

            preds = logits.argmax(1).cpu().tolist()
            labs = batch["label"].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs)
            all_raw.extend(batch["raw_seq"])

    val_loss = val_running_loss / len(dev_loader.dataset)
    pcwa = pattern_complexity_weighted_accuracy(all_raw, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_raw, all_labels, all_preds)
    swa = shape_weighted_accuracy(all_raw, all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (epoch, {"pcwa": pcwa, "cwa": cwa, "swa": swa, "acc": acc})
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"PCWA={pcwa:.4f} CWA={cwa:.4f} SWA={swa:.4f} Acc={acc:.4f}"
    )

# -------------- Save results ---------------
experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
