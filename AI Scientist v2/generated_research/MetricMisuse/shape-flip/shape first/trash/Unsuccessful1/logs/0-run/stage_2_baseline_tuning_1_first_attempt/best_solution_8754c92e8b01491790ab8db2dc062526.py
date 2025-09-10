import os, pathlib, json, random, math, time
from typing import List, Dict
import numpy as np
import torch, datasets
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- dataset helpers (identical to baseline) -------------
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
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


# ---------------------- torch Dataset -------------------------------
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
    ids, labels, seqs = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        if pad_len > 0:
            item_ids = torch.cat(
                [item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]
            )
        else:
            item_ids = item["input_ids"]
        ids.append(item_ids)
        labels.append(item["label"])
        seqs.append(item["seq_text"])
    return {
        "input_ids": torch.stack(ids),
        "label": torch.tensor(labels, dtype=torch.long),
        "seq_text": seqs,
    }


# -------------------- model -----------------------------------------
class BagOfTokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        emb = self.emb(x) * mask
        mean = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(mean)


# ----------------- training / evaluation loops ----------------------
def train_epoch(model, loader, opt, criterion):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        opt.step()
        total += loss.item() * batch["label"].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total = 0.0
    all_preds, all_labels, all_seq = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        total += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().tolist())
        all_seq.extend(batch["seq_text"])
    avg = total / len(loader.dataset)
    return avg, all_seq, all_labels, all_preds


# ========================= main experiment ===========================
def main():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    label_set = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(label_set)}
    idx2label = {i: l for l, i in label2idx.items()}
    print(f"Vocab size: {len(vocab)}  Num classes: {len(label2idx)}")

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate_fn)

    epoch_options = [5, 10, 20, 30]  # hyper-parameter grid
    experiment_data = {"num_epochs": {"SPR_BENCH": {}}}

    for n_epochs in epoch_options:
        print(f"\n===== Training for {n_epochs} epochs =====")
        model = BagOfTokenClassifier(len(vocab), 64, len(label2idx)).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        log_train_loss, log_val_loss, log_val_metric = [], [], []
        best_hwa = -1
        best_state = None

        for ep in range(1, n_epochs + 1):
            tloss = train_epoch(model, train_loader, optim, criterion)
            vloss, seqs, yt, yp = eval_epoch(model, dev_loader, criterion)
            swa = shape_weighted_accuracy(seqs, yt, yp)
            cwa = color_weighted_accuracy(seqs, yt, yp)
            hwa = harmonic_weighted_accuracy(seqs, yt, yp)
            log_train_loss.append(tloss)
            log_val_loss.append(vloss)
            log_val_metric.append({"SWA": swa, "CWA": cwa, "HWA": hwa})
            if hwa > best_hwa:
                best_hwa = hwa
                best_state = model.state_dict()
            print(
                f"  Epoch {ep}/{n_epochs}  train_loss={tloss:.4f}  val_loss={vloss:.4f}  HWA={hwa:.4f}"
            )

        # load best model for testing
        model.load_state_dict(best_state)
        _, seqs_t, yt_t, yp_t = eval_epoch(model, test_loader, criterion)
        swa_t = shape_weighted_accuracy(seqs_t, yt_t, yp_t)
        cwa_t = color_weighted_accuracy(seqs_t, yt_t, yp_t)
        hwa_t = harmonic_weighted_accuracy(seqs_t, yt_t, yp_t)
        print(f"  >> TEST  SWA={swa_t:.4f}  CWA={cwa_t:.4f}  HWA={hwa_t:.4f}")

        # store results
        experiment_data["num_epochs"]["SPR_BENCH"][str(n_epochs)] = {
            "metrics": {
                "train": [],
                "val": log_val_metric,
                "test": {"SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t},
            },
            "losses": {"train": log_train_loss, "val": log_val_loss},
            "predictions": yp_t,
            "ground_truth": yt_t,
        }

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print("\nAll experiments finished and saved to working/experiment_data.npy")


# run
main()
