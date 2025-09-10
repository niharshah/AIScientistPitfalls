import os, pathlib, random, math, time, json
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# mandatory working dir + storage dict
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"weight_decay": {}}  # individual runs inserted later

# ---------------------------------------------------------------------
# device & reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------
# dataset helpers (copied from baseline)
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


# ---------------------------------------------------------------------
# Torch dataset
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
        if pad_len:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(item["label"])
        seq_text.append(item["seq_text"])
    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.tensor(labels, dtype=torch.long),
        "seq_text": seq_text,
    }


# ---------------------------------------------------------------------
# model
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


# ---------------------------------------------------------------------
def train_epoch(model, loader, optim, criterion):
    model.train()
    total = 0
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
        total += loss.item() * batch["label"].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total = 0
    all_seq, y_true, y_pred = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        total += loss.item() * batch["label"].size(0)
        preds = logits.argmax(-1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].cpu().tolist())
        all_seq.extend(batch["seq_text"])
    return total / len(loader.dataset), all_seq, y_true, y_pred


# ---------------------------------------------------------------------
def run_single_experiment(weight_decay: float, data_path: pathlib.Path):
    run_key = f"wd_{weight_decay:.0e}"
    print(f"\n=== Running experiment {run_key} ===")
    spr = load_spr_bench(data_path)
    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    l2i = {l: i for i, l in enumerate(labels)}

    train_ds = SPRTorchDataset(spr["train"], vocab, l2i)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, l2i)
    test_ds = SPRTorchDataset(spr["test"], vocab, l2i)

    loader_args = dict(batch_size=256, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    dev_loader = DataLoader(dev_ds, **loader_args)
    test_loader = DataLoader(test_ds, **loader_args)

    model = BagOfTokenClassifier(len(vocab), 64, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    num_epochs = 5
    best_hwa, best_state = -1, None
    for ep in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optim, criterion)
        val_loss, seqs, y_t, y_p = eval_epoch(model, dev_loader, criterion)
        swa = shape_weighted_accuracy(seqs, y_t, y_p)
        cwa = color_weighted_accuracy(seqs, y_t, y_p)
        hwa = harmonic_weighted_accuracy(seqs, y_t, y_p)

        exp_rec["losses"]["train"].append(tr_loss)
        exp_rec["losses"]["val"].append(val_loss)
        exp_rec["metrics"]["val"].append({"SWA": swa, "CWA": cwa, "HWA": hwa})
        exp_rec["metrics"]["train"].append(None)

        if hwa > best_hwa:
            best_hwa = hwa
            best_state = model.state_dict()

        print(
            f"Epoch {ep} | loss {tr_loss:.4f}/{val_loss:.4f} | SWA {swa:.4f} CWA {cwa:.4f} HWA {hwa:.4f}"
        )

    # load best and test
    model.load_state_dict(best_state)
    _, s_test, y_true_t, y_pred_t = eval_epoch(model, test_loader, criterion)
    swa_t = shape_weighted_accuracy(s_test, y_true_t, y_pred_t)
    cwa_t = color_weighted_accuracy(s_test, y_true_t, y_pred_t)
    hwa_t = harmonic_weighted_accuracy(s_test, y_true_t, y_pred_t)
    print(f"TEST - SWA {swa_t:.4f} CWA {cwa_t:.4f} HWA {hwa_t:.4f}")

    exp_rec["predictions"] = y_pred_t
    exp_rec["ground_truth"] = y_true_t
    exp_rec["metrics"]["test"] = {"SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t}

    return run_key, exp_rec, best_hwa


# ---------------------------------------------------------------------
def main():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    wd_grid = [0.0, 1e-5, 1e-4, 1e-3]
    best_overall, best_key = -1, None

    for wd in wd_grid:
        key, rec, dev_hwa = run_single_experiment(wd, DATA_PATH)
        experiment_data["weight_decay"][key] = rec
        if dev_hwa > best_overall:
            best_overall, best_key = dev_hwa, key

    print(f"\nBest dev HWA achieved by {best_key}: {best_overall:.4f}")

    # save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# execute immediately
main()
