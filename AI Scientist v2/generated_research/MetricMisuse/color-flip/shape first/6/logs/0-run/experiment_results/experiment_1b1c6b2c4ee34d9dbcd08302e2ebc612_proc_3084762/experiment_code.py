import os, random, pathlib, math, time, json, csv
from typing import List, Tuple
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------- experiment dict -------------------
experiment_data = {"learning_rate": {}}

# ---------- working dir & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------- Utility metrics -----------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------------- Data loading -------------------
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def load_csv(split):
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
        data.append({"sequence": seq, "label": rules[i % 2](seq)})
    return data


dataset = {}
for split in ["train", "dev", "test"]:
    rows = load_csv(split)
    if not rows:
        rows = generate_toy(4000 if split == "train" else 1000)
    dataset[split] = rows
print({k: len(v) for k, v in dataset.items()})

# ---------------- Vocabulary ---------------------
tokens = set()
for rows in dataset.values():
    for r in rows:
        tokens.update(r["sequence"].split())
PAD, CLS = "<PAD>", "<CLS>"
itos = [PAD, CLS] + sorted(tokens)
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)
print("Vocab size:", vocab_size)


# -------------- Augmentations --------------------
def aug_sequence(seq: str) -> str:
    toks = seq.split()
    if len(toks) > 1:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    shift = random.randint(0, len(toks) - 1)
    toks = toks[shift:] + toks[:shift]
    return " ".join(toks)


# --------------- Dataset classes -----------------
def encode(seq, max_len=None):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()]
    if max_len:
        ids = ids[:max_len]
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


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
        seq = self.rows[idx]["sequence"]
        return (
            torch.tensor(encode(seq, self.max_len)),
            torch.tensor(self.rows[idx]["label"]),
            seq,
        )


# ---------------- Model --------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, d_model=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, hidden, batch_first=True)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return h.squeeze(0)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(encoder.gru.hidden_size, num_classes)

    def forward(self, x):
        f = self.enc(x)
        return self.fc(f), f


# ------------- Contrastive loss ------------------
def nt_xent(feat, temp=0.5):
    N = feat.shape[0] // 2
    f = F.normalize(feat, dim=1)
    sim = f @ f.T / temp
    sim.masked_fill_(torch.eye(2 * N, device=feat.device).bool(), -9e15)
    targets = torch.arange(N, 2 * N, device=feat.device)
    targets = torch.cat([targets, torch.arange(0, N, device=feat.device)])
    return F.cross_entropy(sim, targets)


# -------------- Hyperparameter sweep -------------
LEARNING_RATES = [3e-4, 1e-3, 3e-3, 1e-2]
BATCH, EPOCH_PRE, EPOCH_FT, max_len = 128, 3, 3, 20
num_classes = len(set(r["label"] for r in dataset["train"]))
contrast_base = SPRContrastive(dataset["train"], max_len)
train_base = SPRLabelled(dataset["train"], max_len)
dev_base = SPRLabelled(dataset["dev"], max_len)
for lr in LEARNING_RATES:
    print(f"\n===== Tuning LR={lr} =====")
    exp_key = str(lr)
    experiment_data["learning_rate"][exp_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "AIS": {"val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # loaders (fresh shuffle each lr)
    contrast_loader = DataLoader(contrast_base, batch_size=BATCH, shuffle=True)
    train_loader = DataLoader(train_base, batch_size=BATCH, shuffle=True)
    dev_loader = DataLoader(dev_base, batch_size=BATCH)
    # model/opt
    encoder = Encoder(vocab_size).to(device)
    model = SPRModel(encoder, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # --------- Phase 1: contrastive ------------
    model.train()
    for ep in range(1, EPOCH_PRE + 1):
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
        print(f"  Pretrain Epoch {ep}: loss={total/len(dataset['train']):.4f}")
    # --------- Phase 2: supervised -------------
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, EPOCH_FT + 1):
        # train
        model.train()
        tr_loss = 0
        for ids, labels, _ in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * ids.size(0)
        tr_loss /= len(dataset["train"])
        # val
        model.eval()
        val_loss, preds, gts, seqs = 0, [], [], []
        with torch.no_grad():
            for ids, labels, seq in dev_loader:
                ids, labels = ids.to(device), labels.to(device)
                logits, _ = model(ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * ids.size(0)
                preds.extend(torch.argmax(logits, 1).cpu().tolist())
                gts.extend(labels.cpu().tolist())
                seqs.extend(seq)
        val_loss /= len(dataset["dev"])
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cwa = color_weighted_accuracy(seqs, gts, preds)

        # AIS
        def compute_ais(rows, n_views=3):
            ok = 0
            with torch.no_grad():
                for r in rows:
                    base = None
                    consistent = True
                    for _ in range(n_views):
                        ids = (
                            torch.tensor(encode(aug_sequence(r["sequence"]), max_len))
                            .unsqueeze(0)
                            .to(device)
                        )
                        pred = torch.argmax(model(ids)[0], 1).item()
                        if base is None:
                            base = pred
                        elif pred != base:
                            consistent = False
                            break
                    if consistent:
                        ok += 1
            return ok / len(rows)

        ais = compute_ais(dataset["dev"])
        # logging
        experiment_data["learning_rate"][exp_key]["metrics"]["train"].append(swa)
        experiment_data["learning_rate"][exp_key]["metrics"]["val"].append(cwa)
        experiment_data["learning_rate"][exp_key]["losses"]["train"].append(tr_loss)
        experiment_data["learning_rate"][exp_key]["losses"]["val"].append(val_loss)
        experiment_data["learning_rate"][exp_key]["AIS"]["val"].append(ais)
        experiment_data["learning_rate"][exp_key]["predictions"] = preds
        experiment_data["learning_rate"][exp_key]["ground_truth"] = gts
        print(
            f"  FT Epoch {ep}: val_loss={val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} AIS={ais:.3f}"
        )

# -------------- Save -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
