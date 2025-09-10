import os, random, pathlib, math, time, json
from typing import List, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- workdir / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- accuracy helpers ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


# ---------- load data ----------
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def load_csv(split):
    import csv, os

    fpath = SPR_PATH / f"{split}.csv"
    rows = []
    if fpath.exists():
        with open(fpath) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({"sequence": r["sequence"], "label": int(r["label"])})
    return rows


def generate_toy(n=2000):
    shapes, colors = "ABC", "123"
    rules = [lambda s: len(s) % 2, lambda s: (s.count("A1") + s.count("B2")) % 3]
    data = []
    for i in range(n):
        seq = " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 8))
        )
        lab = rules[i % 2](seq)
        data.append({"sequence": seq, "label": lab})
    return data


dataset = {}
for split in ["train", "dev", "test"]:
    rows = load_csv(split)
    if not rows:
        rows = generate_toy(4000 if split == "train" else 1000)
    dataset[split] = rows
print({k: len(v) for k, v in dataset.items()})

# ---------- vocab ----------
tokens = set()
for split in dataset.values():
    for r in split:
        tokens.update(r["sequence"].split())
PAD, CLS = "<PAD>", "<CLS>"
itos = [PAD, CLS] + sorted(tokens)
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)
print("Vocab size:", vocab_size)


# ---------- augment / encode ----------
def aug_sequence(seq: str) -> str:
    toks = seq.split()
    if len(toks) > 1:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    shift = random.randint(0, len(toks) - 1)
    toks = toks[shift:] + toks[:shift]
    return " ".join(toks)


def encode(seq, max_len=None):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()]
    if max_len:
        ids = ids[:max_len] + [stoi[PAD]] * (max_len - len(ids))
    return ids


# ---------- datasets ----------
class SPRContrastive(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq = self.rows[idx]["sequence"]
        return torch.tensor(encode(aug_sequence(seq), self.max_len)), torch.tensor(
            encode(aug_sequence(seq), self.max_len)
        )


class SPRLabelled(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(encode(r["sequence"], self.max_len)),
            torch.tensor(r["label"]),
            r["sequence"],
        )


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, vocab, d_model=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, hidden, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return h.squeeze(0)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(self.enc.gru.hidden_size, num_classes)

    def forward(self, x):
        feat = self.enc(x)
        return self.fc(feat), feat


# ---------- contrastive loss ----------
def nt_xent(features, temp=0.5):
    N = features.shape[0] // 2
    f = F.normalize(features, dim=1)
    sim = torch.matmul(f, f.t()) / temp
    sim.masked_fill_(torch.eye(2 * N, device=features.device).bool(), -9e15)
    targets = torch.arange(N, 2 * N, device=features.device)
    targets = torch.cat([targets, torch.arange(0, N, device=features.device)])
    return F.cross_entropy(sim, targets)


# ---------- hyperparameter sweep ----------
BATCH_LIST = [32, 64, 128, 256]
EPOCH_PRE, EPOCH_FT = 3, 3
max_len = 20
num_classes = len(set(r["label"] for r in dataset["train"]))

experiment_data = {"batch_size": {"SPR": {}}}


def compute_ais(rows, model, n_views=3):
    consistent = 0
    with torch.no_grad():
        for r in rows:
            base = None
            ok = True
            for _ in range(n_views):
                ids = (
                    torch.tensor(encode(aug_sequence(r["sequence"]), max_len))
                    .unsqueeze(0)
                    .to(device)
                )
                logit, _ = model(ids)
                pred = torch.argmax(logit, 1).item()
                if base is None:
                    base = pred
                elif pred != base:
                    ok = False
                    break
            if ok:
                consistent += 1
    return consistent / len(rows)


for BATCH in BATCH_LIST:
    print(f"\n===== Training with batch_size={BATCH} =====")
    # dataloaders
    contrast_loader = DataLoader(
        SPRContrastive(dataset["train"], max_len), batch_size=BATCH, shuffle=True
    )
    train_loader = DataLoader(
        SPRLabelled(dataset["train"], max_len), batch_size=BATCH, shuffle=True
    )
    dev_loader = DataLoader(SPRLabelled(dataset["dev"], max_len), batch_size=BATCH)
    # model & optimizer
    encoder = Encoder(vocab_size).to(device)
    model = SPRModel(encoder, num_classes).to(device)
    lr = 1e-3 * (BATCH / 128)  # simple linear scaling
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # storage
    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "AIS": {"val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # ---- contrastive pre-training ----
    for ep in range(1, EPOCH_PRE + 1):
        model.train()
        total = 0
        for v1, v2 in contrast_loader:
            v1, v2 = v1.to(device), v2.to(device)
            _, f1 = model(v1)
            _, f2 = model(v2)
            loss = nt_xent(torch.cat([f1, f2], 0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * v1.size(0)
        print(f"Pretrain Epoch {ep}: loss={total/len(dataset['train']):.4f}")
    # ---- supervised fine-tuning ----
    crit = nn.CrossEntropyLoss()
    for ep in range(1, EPOCH_FT + 1):
        model.train()
        tr_loss = 0
        for ids, labels, _ in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)
            loss = crit(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * ids.size(0)
        tr_loss /= len(dataset["train"])
        model.eval()
        v_loss = 0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for ids, labels, seq in dev_loader:
                ids, labels = ids.to(device), labels.to(device)
                logits, _ = model(ids)
                loss = crit(logits, labels)
                v_loss += loss.item() * ids.size(0)
                preds.extend(torch.argmax(logits, 1).cpu().tolist())
                gts.extend(labels.cpu().tolist())
                seqs.extend(seq)
        v_loss /= len(dataset["dev"])
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        ais = compute_ais(dataset["dev"], model)
        # log
        log["metrics"]["train"].append(swa)
        log["metrics"]["val"].append(cwa)
        log["losses"]["train"].append(tr_loss)
        log["losses"]["val"].append(v_loss)
        log["AIS"]["val"].append(ais)
        log["predictions"] = preds
        log["ground_truth"] = gts
        print(
            f"Epoch {ep}: val_loss={v_loss:.4f} SWA={swa:.3f} CWA={cwa:.3f} AIS={ais:.3f}"
        )
    # store results
    experiment_data["batch_size"]["SPR"][str(BATCH)] = log

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
