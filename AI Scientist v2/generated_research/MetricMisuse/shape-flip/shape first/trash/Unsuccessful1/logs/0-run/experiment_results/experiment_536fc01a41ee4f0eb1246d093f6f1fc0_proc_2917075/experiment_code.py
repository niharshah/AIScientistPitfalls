import os, pathlib, math, time, json, random
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------- misc paths -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- dataset helpers -------------
from datasets import load_dataset, DatasetDict


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


# ---------------- torch dataset ---------------
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
        }


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


def collate_fn(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])
    input_ids, labels, seq_text = [], [], []
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


# ------------------ model ---------------------
class BagOfTokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        emb = self.emb(x) * mask
        mean = emb.sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean)


# ------------- train / eval helpers -----------
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
    all_preds, all_labels, all_seq = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().tolist())
        all_seq.extend(batch["seq_text"])
    return (
        total_loss / len(loader.dataset),
        all_seq,
        all_labels,
        all_preds,
    )


# ================================================================
def main():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    label_set = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(label_set)}

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    learning_rates = [3e-4, 1e-3, 3e-3]
    num_epochs = 5

    experiment_data = {"lr_sweep": {}}
    best_lr, best_dev_hwa = None, -1.0

    for lr in learning_rates:
        print(f"\n==== Training with lr={lr} ====")
        model = BagOfTokenClassifier(len(vocab), 64, len(label2idx)).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        entry = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(1, num_epochs + 1):
            tr_loss = train_epoch(model, train_loader, optim, criterion)
            val_loss, seqs, y_true, y_pred = eval_epoch(model, dev_loader, criterion)
            swa = shape_weighted_accuracy(seqs, y_true, y_pred)
            cwa = color_weighted_accuracy(seqs, y_true, y_pred)
            hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)

            entry["losses"]["train"].append(tr_loss)
            entry["losses"]["val"].append(val_loss)
            entry["metrics"]["train"].append(None)
            entry["metrics"]["val"].append({"SWA": swa, "CWA": cwa, "HWA": hwa})

            print(
                f"  Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} HWA={hwa:.4f}"
            )

        # use last epoch dev HWA for model selection
        dev_hwa_last = entry["metrics"]["val"][-1]["HWA"]
        if dev_hwa_last > best_dev_hwa:
            best_dev_hwa = dev_hwa_last
            best_lr = lr
            # evaluate on test to store best predictions/gt
            _, seqs_t, y_true_t, y_pred_t = eval_epoch(model, test_loader, criterion)
            best_test = {
                "SWA": shape_weighted_accuracy(seqs_t, y_true_t, y_pred_t),
                "CWA": color_weighted_accuracy(seqs_t, y_true_t, y_pred_t),
                "HWA": harmonic_weighted_accuracy(seqs_t, y_true_t, y_pred_t),
            }
            best_preds, best_truth = y_pred_t, y_true_t

        experiment_data["lr_sweep"][str(lr)] = entry

    # store best lr info
    experiment_data["lr_sweep"]["best_lr"] = best_lr
    experiment_data["lr_sweep"]["best_dev_hwa"] = best_dev_hwa
    experiment_data["lr_sweep"]["best_test_metrics"] = best_test
    experiment_data["lr_sweep"]["best_test_predictions"] = best_preds
    experiment_data["lr_sweep"]["best_test_ground_truth"] = best_truth

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print(f"\nBest lr={best_lr} with dev HWA={best_dev_hwa:.4f}")
    print(f"Test metrics at best lr: {best_test}")


# run immediately
main()
