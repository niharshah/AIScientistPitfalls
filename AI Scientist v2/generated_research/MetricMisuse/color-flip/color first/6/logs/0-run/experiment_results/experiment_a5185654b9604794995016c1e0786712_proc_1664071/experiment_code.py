import os, math, pathlib, random, json, time, warnings
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------ metrics ----------------------------------------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def entropy_weight(seq):
    toks = seq.strip().split()
    if not toks:
        return 0.0
    freqs, total = Counter(toks), len(toks)
    return -sum((c / total) * math.log2(c / total) for c in freqs.values())


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def ewa(seqs, y_true, y_pred):
    w = [entropy_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ------------------------ load SPR_BENCH ---------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DEFAULT_PATHS = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]
for p in DEFAULT_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder missing.")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------------------ vocab / labels ---------------------------------
def build_vocab(seqs, min_freq=1):
    cnt = Counter()
    [cnt.update(s.strip().split()) for s in seqs]
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in cnt.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print("Vocab size:", len(vocab), "| Num labels:", len(label2idx))


# ------------------------ Torch Dataset ----------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, vocab, label2idx):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]
        self.vocab = vocab
        self.label2idx = label2idx

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = [self.vocab.get(t, 1) for t in self.seqs[idx].strip().split()]
        return {
            "input_ids": torch.tensor(toks),
            "length": len(toks),
            "label": self.label2idx[self.labels[idx]],
            "seq_raw": self.seqs[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch)
    pad = 0
    ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    lengths, labels, seq_raw = [], [], []
    for i, item in enumerate(batch):
        l = item["length"]
        ids[i, :l] = item["input_ids"]
        lengths.append(l)
        labels.append(item["label"])
        seq_raw.append(item["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.tensor(lengths),
        "labels": torch.tensor(labels),
        "seq_raw": seq_raw,
    }


train_base = SPRTorchDataset(spr["train"], vocab, label2idx)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)


# ------------------------ Model ------------------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_labels)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        x = x * mask
        summed = x.sum(1)
        lengths = lengths.unsqueeze(1).type_as(summed)
        return self.fc(summed / lengths.clamp(min=1))


# ------------------------ hyperparameter sweep ---------------------------
BATCH_SIZES = [16, 32, 64, 128]
EPOCHS = 5
experiment_data = {"batch_size_tuning": {"SPR_BENCH": {}}}

for bs in BATCH_SIZES:
    print(f"\n=== Training with batch_size={bs} ===")
    # fresh splits to ensure shuffle difference negligible
    train_loader = DataLoader(
        train_base, batch_size=bs, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)
    model = MeanEmbedClassifier(len(vocab), 64, len(label2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    run_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    best_val = None
    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tot_loss = 0
        n = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            labs = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(ids, lens)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        tr_loss = tot_loss / n
        run_dict["losses"]["train"].append((epoch, tr_loss))
        # validate
        model.eval()
        val_loss = 0
        n = 0
        all_seq, all_true, all_pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                lens = batch["lengths"].to(device)
                labs = batch["labels"].to(device)
                logits = model(ids, lens)
                loss = criterion(logits, labs)
                val_loss += loss.item() * ids.size(0)
                n += ids.size(0)
                preds = logits.argmax(1).cpu().tolist()
                labels = labs.cpu().tolist()
                all_seq.extend(batch["seq_raw"])
                all_true.extend([idx2label[i] for i in labels])
                all_pred.extend([idx2label[i] for i in preds])
        val_loss /= n
        run_dict["losses"]["val"].append((epoch, val_loss))
        cwa_s, swa_s, ewa_s = (
            cwa(all_seq, all_true, all_pred),
            swa(all_seq, all_true, all_pred),
            ewa(all_seq, all_true, all_pred),
        )
        run_dict["metrics"]["val"].append(
            (epoch, {"CWA": cwa_s, "SWA": swa_s, "EWA": ewa_s})
        )
        print(
            f"Epoch {epoch} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} CWA={cwa_s:.3f} SWA={swa_s:.3f} EWA={ewa_s:.3f}"
        )
    # ---------------- test evaluation ------------------------------------
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(ids, lens)
            preds = logits.argmax(1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_seq.extend(batch["seq_raw"])
            all_true.extend([idx2label[i] for i in labels])
            all_pred.extend([idx2label[i] for i in preds])
    run_dict["predictions"], run_dict["ground_truth"] = all_pred, all_true
    test_cwa, test_swa, test_ewa = (
        cwa(all_seq, all_true, all_pred),
        swa(all_seq, all_true, all_pred),
        ewa(all_seq, all_true, all_pred),
    )
    run_dict["metrics"]["test"] = {"CWA": test_cwa, "SWA": test_swa, "EWA": test_ewa}
    print(f"Test metrics | CWA={test_cwa:.3f} SWA={test_swa:.3f} EWA={test_ewa:.3f}")
    # store
    experiment_data["batch_size_tuning"]["SPR_BENCH"][f"bs_{bs}"] = run_dict

# ---------------- save & optional plot -----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
