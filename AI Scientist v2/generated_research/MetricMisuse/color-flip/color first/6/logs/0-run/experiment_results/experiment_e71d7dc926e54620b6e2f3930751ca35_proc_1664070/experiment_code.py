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

import os, math, time, json, random, pathlib
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

# ------------- basic cfg / reproducibility ------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- load SPR_BENCH --------------------------------
def load_spr_bench(root: pathlib.Path):
    def _l(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return {"train": _l("train.csv"), "dev": _l("dev.csv"), "test": _l("test.csv")}


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH not found.")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------- metrics --------------------------------------
def _color_var(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def _shape_var(seq):
    return len({tok[0] for tok in seq.split() if tok})


def _ent(seq):
    toks = seq.split()
    n = len(toks)
    if n == 0:
        return 0.0
    from collections import Counter

    freqs = Counter(toks)
    return -sum(c / n * math.log2(c / n) for c in freqs.values())


def cwa(s, y, p):
    w = [_color_var(i) for i in s]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0


def swa(s, y, p):
    w = [_shape_var(i) for i in s]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0


def ewa(s, y, p):
    w = [_ent(i) for i in s]
    c = [wt if t == q else 0 for wt, t, q in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0


# ------------- vocab / labels --------------------------------
def build_vocab(seqs, min_freq=1):
    from collections import Counter

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
num_labels = len(label2idx)
print(f"Vocab={len(vocab)}, Labels={num_labels}")


# ------------- dataset / loader ------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf, vocab, l2i):
        self.seq = hf["sequence"]
        self.lab = hf["label"]
        self.vocab = vocab
        self.l2i = l2i

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        ids = [self.vocab.get(t, 1) for t in self.seq[idx].split()]
        return {
            "input_ids": torch.tensor(ids),
            "length": len(ids),
            "label": self.l2i[self.lab[idx]],
            "seq_raw": self.seq[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch)
    pad = 0
    ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    lengths, labels, raw = [], [], []
    for i, b in enumerate(batch):
        l = b["length"]
        ids[i, :l] = b["input_ids"]
        lengths.append(l)
        labels.append(b["label"])
        raw.append(b["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.tensor(lengths),
        "labels": torch.tensor(labels),
        "seq_raw": raw,
    }


train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)
train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ------------- model -----------------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vsz, edim, nlbl):
        super().__init__()
        self.emb = nn.Embedding(vsz, edim, padding_idx=0)
        self.fc = nn.Linear(edim, nlbl)

    def forward(self, ids, lens):
        x = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        mean = (x * mask).sum(1) / lens.unsqueeze(1).clamp(min=1).type_as(x)
        return self.fc(mean)


# ------------- hyperparam tuning over beta2 ------------------
beta2_values = [0.95, 0.97, 0.98, 0.99, 0.999]
EPOCHS = 5
experiment_data = {"adam_beta2": {}}

for beta2 in beta2_values:
    print(f"\n===== Training with beta2={beta2} =====")
    model = MeanEmbedClassifier(len(vocab), 64, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, beta2))

    exp_key = str(beta2)
    experiment_data["adam_beta2"][exp_key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tot_loss = 0
        n = 0
        for batch in train_loader(64):
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            labs = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lens), labs)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        tr_loss = tot_loss / n
        experiment_data["adam_beta2"][exp_key]["SPR_BENCH"]["losses"]["train"].append(
            (epoch, tr_loss)
        )

        # ---- validate ----
        model.eval()
        val_loss = 0
        n = 0
        seqs, true, pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                lens = batch["lengths"].to(device)
                labs = batch["labels"].to(device)
                logits = model(ids, lens)
                loss = criterion(logits, labs)
                val_loss += loss.item() * ids.size(0)
                n += ids.size(0)
                pr = logits.argmax(1).cpu().tolist()
                la = labs.cpu().tolist()
                seqs.extend(batch["seq_raw"])
                true.extend([idx2label[i] for i in la])
                pred.extend([idx2label[i] for i in pr])
        val_loss /= n
        experiment_data["adam_beta2"][exp_key]["SPR_BENCH"]["losses"]["val"].append(
            (epoch, val_loss)
        )
        cwa_s, swa_s, ewa_s = (
            cwa(seqs, true, pred),
            swa(seqs, true, pred),
            ewa(seqs, true, pred),
        )
        experiment_data["adam_beta2"][exp_key]["SPR_BENCH"]["metrics"]["val"].append(
            (epoch, {"CWA": cwa_s, "SWA": swa_s, "EWA": ewa_s})
        )
        print(
            f"Epoch {epoch} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | CWA {cwa_s:.4f} | SWA {swa_s:.4f} | EWA {ewa_s:.4f}"
        )

    # ---- final test ----
    model.eval()
    seqs, true, pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(ids, lens)
            pr = logits.argmax(1).cpu().tolist()
            la = batch["labels"].cpu().tolist()
            seqs.extend(batch["seq_raw"])
            true.extend([idx2label[i] for i in la])
            pred.extend([idx2label[i] for i in pr])
    experiment_data["adam_beta2"][exp_key]["SPR_BENCH"]["predictions"] = pred
    experiment_data["adam_beta2"][exp_key]["SPR_BENCH"]["ground_truth"] = true
    tcwa, tswa, tewa = (
        cwa(seqs, true, pred),
        swa(seqs, true, pred),
        ewa(seqs, true, pred),
    )
    print(f"Test CWA {tcwa:.4f} | SWA {tswa:.4f} | EWA {tewa:.4f}")

    # ---- plot losses ----
    tr = [
        l
        for _, l in experiment_data["adam_beta2"][exp_key]["SPR_BENCH"]["losses"][
            "train"
        ]
    ]
    vl = [
        l
        for _, l in experiment_data["adam_beta2"][exp_key]["SPR_BENCH"]["losses"]["val"]
    ]
    ep = range(1, EPOCHS + 1)
    plt.figure()
    plt.plot(ep, tr, label="train")
    plt.plot(ep, vl, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss (beta2={beta2})")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_beta2_{beta2}.png"))
    plt.close()

# ------------- save experiment data --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiments finished and saved.")
