import os, pathlib, random, math, time, json
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ==================  mandatory working dir & device ==================
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================   dataset helper functions  ======================
from datasets import load_dataset, DatasetDict  # lightweight


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):  # helper for train/dev/test csvs
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ("train", "dev", "test"):
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


# ====================   Torch dataset wrappers   =====================
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], label2idx: Dict[str, int]):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [
            self.vocab.get(tok, self.vocab["<unk>"]) for tok in self.seqs[idx].split()
        ]
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
        if len(ids) < max_len:
            ids = torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)])
        input_ids.append(ids)
        labels.append(item["label"])
        seq_text.append(item["seq_text"])
    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.stack(labels),
        "seq_text": seq_text,
    }


# ===========================  model  =================================
class BagOfTokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        emb = self.emb(x) * mask
        summed = emb.sum(1)
        lengths = mask.sum(1).clamp(min=1)
        mean = summed / lengths
        return self.fc(mean)


# ====================   train / eval loops   =========================
def train_epoch(model, loader, opt, criterion):
    model.train()
    tot = 0.0
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
        tot += loss.item() * batch["label"].size(0)
    return tot / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    tot = 0.0
    all_p, all_l, all_s = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        tot += loss.item() * batch["label"].size(0)
        all_p.extend(logits.argmax(-1).cpu().tolist())
        all_l.extend(batch["label"].cpu().tolist())
        all_s.extend(batch["seq_text"])
    return tot / len(loader.dataset), all_s, all_l, all_p


# ===========================  main  ==================================
def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(labels)}
    print(f"Vocab size {len(vocab)}, classes {len(labels)}")

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)
    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate_fn)

    embedding_dims = [32, 64, 128, 256]
    num_epochs = 5
    experiment_data = {"embedding_dim_tuning": {}}

    for emb_dim in embedding_dims:
        print(f"\n========= Training with embedding_dim {emb_dim} =========")
        model = BagOfTokenClassifier(len(vocab), emb_dim, len(labels)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        key = f"SPR_BENCH_emb{emb_dim}"
        experiment_data["embedding_dim_tuning"][key] = {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(1, num_epochs + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, seqs, y_true, y_pred = eval_epoch(model, dev_loader, criterion)
            swa = shape_weighted_accuracy(seqs, y_true, y_pred)
            cwa = color_weighted_accuracy(seqs, y_true, y_pred)
            hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)

            d = experiment_data["embedding_dim_tuning"][key]
            d["losses"]["train"].append(tr_loss)
            d["losses"]["val"].append(val_loss)
            d["metrics"]["train"].append(None)
            d["metrics"]["val"].append({"SWA": swa, "CWA": cwa, "HWA": hwa})

            print(
                f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
            )

        # final test eval
        _, seqs_t, y_true_t, y_pred_t = eval_epoch(model, test_loader, criterion)
        swa_t = shape_weighted_accuracy(seqs_t, y_true_t, y_pred_t)
        cwa_t = color_weighted_accuracy(seqs_t, y_true_t, y_pred_t)
        hwa_t = harmonic_weighted_accuracy(seqs_t, y_true_t, y_pred_t)
        print(
            f"TEST (emb={emb_dim})  SWA={swa_t:.4f}  CWA={cwa_t:.4f}  HWA={hwa_t:.4f}"
        )

        d = experiment_data["embedding_dim_tuning"][key]
        d["metrics"]["test"] = {"SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t}
        d["predictions"] = y_pred_t
        d["ground_truth"] = y_true_t

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# run
main()
