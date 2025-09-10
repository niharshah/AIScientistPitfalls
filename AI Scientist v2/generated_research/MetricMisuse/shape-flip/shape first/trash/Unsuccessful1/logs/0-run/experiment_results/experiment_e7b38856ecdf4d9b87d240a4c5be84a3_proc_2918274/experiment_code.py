import os, pathlib, math, time, json, random
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- paths / dirs -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- device ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- dataset helpers --------------
from datasets import load_dataset, DatasetDict  # lightweight


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


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


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0


# -------------------- torch dataset ------------
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


def build_vocab(train_sequences, min_freq: int = 1):
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
        if (pad := max_len - len(ids)) > 0:
            ids = torch.cat([ids, torch.zeros(pad, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(item["label"])
        seq_text.append(item["seq_text"])
    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.tensor(labels, dtype=torch.long),
        "seq_text": seq_text,
    }


# -------------------- model --------------------
class BagOfTokenClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        emb = self.emb(x) * mask
        mean = emb.sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean)


# ------------------ train / eval ---------------
def train_epoch(model, loader, optim, crit):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optim.zero_grad()
        loss = crit(model(batch["input_ids"]), batch["label"])
        loss.backward()
        optim.step()
        total += loss.item() * batch["label"].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, crit):
    model.eval()
    total, preds, labels, seqs = 0.0, [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = crit(logits, batch["label"])
        total += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["seq_text"])
    return total / len(loader.dataset), seqs, labels, preds


# ------------------- main ----------------------
def run_batchsize_tuning():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    label2idx = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    batch_sizes = [64, 128, 256, 512]
    num_epochs = 5
    experiment_data = {"batch_size_tuning": {}}

    for bs in batch_sizes:
        tag = f"SPR_BENCH_bs{bs}"
        print(f"\n===== Training with batch_size={bs} =====")
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
        )
        dev_loader = DataLoader(dev_ds, batch_size=bs, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=bs, collate_fn=collate_fn)

        model = BagOfTokenClassifier(len(vocab), 64, len(label2idx)).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        run_dict = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(1, num_epochs + 1):
            tr_loss = train_epoch(model, train_loader, optim, crit)
            val_loss, seqs, y_true, y_pred = eval_epoch(model, dev_loader, crit)
            swa = shape_weighted_accuracy(seqs, y_true, y_pred)
            cwa = color_weighted_accuracy(seqs, y_true, y_pred)
            hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)

            run_dict["losses"]["train"].append(tr_loss)
            run_dict["losses"]["val"].append(val_loss)
            run_dict["metrics"]["train"].append(None)
            run_dict["metrics"]["val"].append({"SWA": swa, "CWA": cwa, "HWA": hwa})

            print(
                f"Epoch {epoch}/{num_epochs}  train_loss={tr_loss:.4f}  "
                f"val_loss={val_loss:.4f}  SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}"
            )

        # final test
        _, s_t, y_t, y_p = eval_epoch(model, test_loader, crit)
        swa_t = shape_weighted_accuracy(s_t, y_t, y_p)
        cwa_t = color_weighted_accuracy(s_t, y_t, y_p)
        hwa_t = harmonic_weighted_accuracy(s_t, y_t, y_p)
        print(f"TEST  SWA={swa_t:.4f}  CWA={cwa_t:.4f}  HWA={hwa_t:.4f}")

        run_dict["predictions"] = y_p
        run_dict["ground_truth"] = y_t
        run_dict["metrics"]["test"] = {"SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t}

        experiment_data["batch_size_tuning"][tag] = run_dict

    # save all results
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# -------------- execute ------------------------
run_batchsize_tuning()
