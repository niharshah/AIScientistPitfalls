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

import os, pathlib, math, time, json, random
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ======================================================================
# GPU / CPU handling (obligatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ======================================================================

# ------------ dataset helpers  (copied from given SPR.py snippet) -----------------
from datasets import load_dataset, DatasetDict  # lightweight, no pandas


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


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


# ---------------------------------------------------------------------


# ----------------------- torch Dataset --------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], label2idx: Dict[str, int]):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = self.seqs[idx].split()
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": self.seqs[idx],
        }  # keep original for metrics


def build_vocab(train_sequences: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq = {}
    for s in train_sequences:
        for tok in s.split():
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, cnt in freq.items():
        if cnt >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


# ----------------------------------------------------------------------


def collate_fn(batch):
    # sort by len for efficiency
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])
    input_ids = []
    labels = []
    seq_text = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(item["label"])
        seq_text.append(item["seq_text"])
    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.tensor(labels, dtype=torch.long),
        "seq_text": seq_text,
    }


# --------------------------- model ------------------------------------
class BagOfTokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B,T]
        mask = (x != 0).unsqueeze(-1)  # ignore pad
        emb = self.emb(x)
        emb = emb * mask
        summed = emb.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean = summed / lengths
        return self.fc(mean)


# ----------------------------------------------------------------------


def train_epoch(model, loader, optim, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optim.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optim.step()
        total_loss += loss.item() * batch["label"].size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_seq = []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().tolist())
        all_seq.extend(batch["seq_text"])
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_seq, all_labels, all_preds


# ======================================================================
def main_training():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    label_set = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(label_set)}
    idx2label = {i: l for l, i in label2idx.items()}
    print(f"Vocab size: {len(vocab)}, num_classes: {len(label2idx)}")

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate_fn)

    model = BagOfTokenClassifier(len(vocab), 64, len(label2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    experiment_data = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, criterion)
        val_loss, seqs, y_true, y_pred = eval_epoch(model, dev_loader, criterion)
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)

        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(
            None
        )  # no train metric for now
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            {"SWA": swa, "CWA": cwa, "HWA": hwa}
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}"
        )

    # final test evaluation
    _, seqs_t, y_true_t, y_pred_t = eval_epoch(model, test_loader, criterion)
    swa_t = shape_weighted_accuracy(seqs_t, y_true_t, y_pred_t)
    cwa_t = color_weighted_accuracy(seqs_t, y_true_t, y_pred_t)
    hwa_t = harmonic_weighted_accuracy(seqs_t, y_true_t, y_pred_t)
    print(f"\nTEST  SWA={swa_t:.4f}  CWA={cwa_t:.4f}  HWA={hwa_t:.4f}")

    experiment_data["SPR_BENCH"]["predictions"] = y_pred_t
    experiment_data["SPR_BENCH"]["ground_truth"] = y_true_t
    experiment_data["SPR_BENCH"]["metrics"]["test"] = {
        "SWA": swa_t,
        "CWA": cwa_t,
        "HWA": hwa_t,
    }

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    # torch.save(model.state_dict(), os.path.join(working_dir, 'baseline_bagofemb.pt'))


# run immediately (no if __main__)
main_training()
