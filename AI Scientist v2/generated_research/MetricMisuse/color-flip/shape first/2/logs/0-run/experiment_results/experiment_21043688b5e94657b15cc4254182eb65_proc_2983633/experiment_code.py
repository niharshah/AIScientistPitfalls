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

import os, pathlib, random, time, json, math
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment container ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- locate SPR_BENCH ----------
def find_spr_bench_path() -> pathlib.Path:
    """Return absolute Path to SPR_BENCH containing the expected csv files."""
    candidates = [
        os.environ.get("SPR_BENCH_PATH", ""),  # user-specified
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",  # fallback (from error log)
    ]
    for c in candidates:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print(f"Found SPR_BENCH at: {p}")
            return p
    raise FileNotFoundError(
        "SPR_BENCH directory with train.csv/dev.csv/test.csv not found. "
        "Set env var SPR_BENCH_PATH or place directory next to this script."
    )


DATA_PATH = find_spr_bench_path()


# ---------- dataset utilities ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",  # read full csv
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ---------- load dataset ----------
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- build vocab / labels ----------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(dataset):
    labels = sorted({ex["label"] for ex in dataset})
    return {lab: i for i, lab in enumerate(labels)}


vocab = build_vocab(spr["train"])
label2id = build_label_map(spr["train"])
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
pad_id = vocab["<pad>"]
print(f"Vocab size = {len(vocab)}, num_labels = {num_labels}")


# ---------- Torch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, label2id):
        self.data = hf_dataset
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def encode_seq(self, seq):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __getitem__(self, idx):
        ex = self.data[idx]
        ids = self.encode_seq(ex["sequence"])
        label = self.label2id[ex["label"]]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    sequences = []
    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i] = b["label"]
        sequences.append(b["sequence"])
    return {"input_ids": input_ids, "labels": labels, "sequences": sequences}


train_ds = SPRTorchDataset(spr["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id)

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0
)  # <= BUGFIX
dev_loader = DataLoader(
    dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
)


# ---------- model ----------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.emb(x)  # [B, L, E]
        outputs, _ = self.lstm(emb)  # [B, L, 2H]
        mask = (x != pad_id).unsqueeze(-1)  # [B, L, 1]
        summed = (outputs * mask).sum(1)  # [B, 2H]
        lengths = mask.sum(1).clamp(min=1)  # [B,1]
        mean = summed / lengths
        return self.fc(mean)  # [B, C]


model = BiLSTMClassifier(len(vocab), 64, 128, num_labels, pad_idx=pad_id).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- training ----------
epochs = 5
for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    running_loss = 0.0
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
    train_loss = running_loss / len(train_ds)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"epoch": epoch, "loss": train_loss}
    )

    # ---- eval ----
    model.eval()
    val_loss, all_pred, all_true, all_seq = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            tensor_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(tensor_batch["input_ids"])
            loss = criterion(logits, tensor_batch["labels"])
            val_loss += loss.item() * tensor_batch["labels"].size(0)
            preds = logits.argmax(-1).cpu().tolist()
            truths = tensor_batch["labels"].cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(truths)
            all_seq.extend(batch["sequences"])
    val_loss /= len(dev_ds)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "swa": swa, "cwa": cwa, "hwa": hwa, "loss": val_loss}
    )
    experiment_data["SPR_BENCH"]["predictions"] = all_pred
    experiment_data["SPR_BENCH"]["ground_truth"] = all_true

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
        f"| SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
    )

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
