import os, math, time, json, random, pathlib
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

# ---------- experiment data dict ----------
experiment_data = {"label_smoothing": {}}

# ---------- working dir / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper : load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path):
    def _load(csv_name):  # tiny helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


DEFAULT_PATHS = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]
for p in DEFAULT_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found.")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- metrics ----------
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def entropy_weight(seq):
    toks = seq.split()
    n = len(toks)
    if n == 0:
        return 0.0
    freq = Counter(toks)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _weighted_acc(seqs, y_true, y_pred, weight_fun):
    w = [weight_fun(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def cwa(s, y_t, y_p):
    return _weighted_acc(s, y_t, y_p, count_color_variety)


def swa(s, y_t, y_p):
    return _weighted_acc(s, y_t, y_p, count_shape_variety)


def ewa(s, y_t, y_p):
    return _weighted_acc(s, y_t, y_p, entropy_weight)


# ---------- vocab / label mapping ----------
def build_vocab(seqs, min_freq=1):
    cnt = Counter()
    [cnt.update(s.split()) for s in seqs]
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in cnt.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, ds, vocab, l2i):
        self.seqs, self.labels = ds["sequence"], ds["label"]
        self.vocab, self.l2i = vocab, l2i

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = [self.vocab.get(t, 1) for t in self.seqs[idx].split()]
        return {
            "input_ids": torch.tensor(toks),
            "length": len(toks),
            "label": self.l2i[self.labels[idx]],
            "seq_raw": self.seqs[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch)
    pad = 0
    ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    lengths, labels, seqs = [], [], []
    for i, b in enumerate(batch):
        l = b["length"]
        ids[i, :l] = b["input_ids"]
        lengths.append(l)
        labels.append(b["label"])
        seqs.append(b["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.tensor(lengths),
        "labels": torch.tensor(labels),
        "seq_raw": seqs,
    }


train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- model ----------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_labels)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        summed = (x * mask).sum(1)
        mean = summed / lengths.unsqueeze(1).type_as(summed).clamp(min=1)
        return self.fc(mean)


# ---------- hyper-parameter sweep ----------
EPSILONS = [0.00, 0.05, 0.10, 0.20]
EPOCHS = 5

for eps in EPSILONS:
    exp_key = f"SPR_BENCH_eps{eps}"
    experiment_data["label_smoothing"][exp_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = MeanEmbedClassifier(len(vocab), 64, len(label2idx)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tot_loss = n = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            lab = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lens), lab)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        tr_loss = tot_loss / n
        experiment_data["label_smoothing"][exp_key]["losses"]["train"].append(
            (epoch, tr_loss)
        )

        # ---- validate ----
        model.eval()
        val_loss = n = 0
        seqs, true, pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                lens = batch["lengths"].to(device)
                lab = batch["labels"].to(device)
                logits = model(ids, lens)
                loss = criterion(logits, lab)
                val_loss += loss.item() * ids.size(0)
                n += ids.size(0)
                p = logits.argmax(1).cpu().tolist()
                l = lab.cpu().tolist()
                seqs.extend(batch["seq_raw"])
                true.extend([idx2label[i] for i in l])
                pred.extend([idx2label[i] for i in p])
        val_loss /= n
        experiment_data["label_smoothing"][exp_key]["losses"]["val"].append(
            (epoch, val_loss)
        )
        cwa_s, swa_s, ewa_s = (
            cwa(seqs, true, pred),
            swa(seqs, true, pred),
            ewa(seqs, true, pred),
        )
        experiment_data["label_smoothing"][exp_key]["metrics"]["val"].append(
            (epoch, {"CWA": cwa_s, "SWA": swa_s, "EWA": ewa_s})
        )
        print(
            f"[eps={eps}] Epoch {epoch}: train_loss={tr_loss:.4f}, "
            f"val_loss={val_loss:.4f}, CWA={cwa_s:.4f}, SWA={swa_s:.4f}, EWA={ewa_s:.4f}"
        )

    # ---------- test evaluation ----------
    model.eval()
    seqs, true, pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(ids, lens)
            p = logits.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            seqs.extend(batch["seq_raw"])
            true.extend([idx2label[i] for i in l])
            pred.extend([idx2label[i] for i in p])
    experiment_data["label_smoothing"][exp_key]["predictions"] = pred
    experiment_data["label_smoothing"][exp_key]["ground_truth"] = true
    tcwa, tswa, tewa = (
        cwa(seqs, true, pred),
        swa(seqs, true, pred),
        ewa(seqs, true, pred),
    )
    print(f"[eps={eps}] Test  CWA={tcwa:.4f}, SWA={tswa:.4f}, EWA={tewa:.4f}")

    # ---------- plot losses ----------
    epochs = [
        e for e, _ in experiment_data["label_smoothing"][exp_key]["losses"]["train"]
    ]
    tr = [l for _, l in experiment_data["label_smoothing"][exp_key]["losses"]["train"]]
    vl = [l for _, l in experiment_data["label_smoothing"][exp_key]["losses"]["val"]]
    plt.figure()
    plt.plot(epochs, tr, label="train")
    plt.plot(epochs, vl, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss ε={eps}")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_eps{eps}.png"))
    plt.close()

# ---------- save all experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiments finished and saved.")
