# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, math, time, pathlib, itertools, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- try loading SPR_BENCH -------
def try_load_real_dataset():
    try:
        from SPR import load_spr_bench  # provided by grader

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        if not DATA_PATH.exists():
            DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH.")
        return dset
    except Exception as e:
        print("Could not load real SPR_BENCH:", e)
        return None


real_dset = try_load_real_dataset()


# -------------- synthetic fallback ------------
def make_random_token():
    shapes = ["R", "S", "T", "U", "V"]
    colors = ["A", "B", "C", "D", "E"]
    return random.choice(shapes) + random.choice(colors)


def generate_sequence(min_len=3, max_len=10):
    return " ".join(
        make_random_token() for _ in range(random.randint(min_len, max_len))
    )


def generate_synthetic_split(n_rows: int):
    return [
        {"id": i, "sequence": generate_sequence(), "label": random.randint(0, 3)}
        for i in range(n_rows)
    ]


if real_dset is None:
    print("Generating synthetic data …")
    real_dset = {
        "train": generate_synthetic_split(1000),
        "dev": generate_synthetic_split(200),
        "test": generate_synthetic_split(200),
    }


# -------------- SCWA helpers ------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split()))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split()))


def scwa_metric(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# --------------- vocab ------------------------
PAD, TUNK, TMASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(dataset):
    vocab = set()
    for row in dataset:
        vocab.update(row["sequence"].split())
    vocab_list = [PAD, TUNK, TMASK] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(vocab_list)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size = len(stoi)
print("vocab size:", vocab_size)


def encode(seq: str, max_len: int):
    ids = [stoi.get(tok, stoi[TUNK]) for tok in seq.split()][:max_len]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


MAX_LEN = 20


# --------------- datasets ---------------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN, supervised=False):
        self.rows, self.max_len, self.supervised = rows, max_len, supervised

    def augment(self, tokens: List[int]):
        toks = [t for t in tokens if t != stoi[PAD]]
        if not toks:
            toks = [stoi[PAD]]
        if random.random() < 0.3:
            del toks[random.randrange(len(toks))]
        if len(toks) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(toks)), 2)
            toks[i], toks[j] = toks[j], toks[i]
        toks = [stoi[TMASK] if random.random() < 0.15 else t for t in toks]
        return encode(" ".join(itos[t] for t in toks), self.max_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = encode(row["sequence"], self.max_len)
        v1, v2 = torch.tensor(self.augment(ids)), torch.tensor(self.augment(ids))
        if self.supervised:
            return {
                "view1": v1,
                "view2": v2,
                "label": torch.tensor(row["label"]),
                "seq": row["sequence"],
            }
        return {"view1": v1, "view2": v2, "seq": row["sequence"]}


class SPRSupervisedDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "ids": torch.tensor(encode(row["sequence"], self.max_len)),
            "label": torch.tensor(row["label"]),
            "seq": row["sequence"],
        }


# --------------- model ------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden * 2, 128)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


# -------------- loss --------------------------
def simclr_loss(z1, z2, temperature=0.5):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], 0)
    logits = torch.matmul(reps, reps.T) / temperature
    logits.fill_diagonal_(-9e15)
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z1.device)
    return nn.functional.cross_entropy(logits, pos)


# -------------- eval --------------------------
def evaluate(model, dl, criterion):
    model.eval()
    preds, labels, seqs = [], [], []
    loss_sum, cnt = 0, 0
    with torch.no_grad():
        for batch in dl:
            ids = batch["ids"].to(device)
            y = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, y)
            loss_sum += loss.item()
            cnt += 1
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            seqs.extend(batch["seq"])
    return loss_sum / cnt, scwa_metric(seqs, labels, preds), preds, labels, seqs


# -------------- experiment dict ---------------
experiment_data = {"BATCH_SIZE": {"SPR_BENCH": {}}}

# -------------- hyperparam sweep --------------
BATCH_SIZES = [64, 128, 256]
PRE_EPOCHS, FT_EPOCHS = 3, 5
NUM_CLASSES = len(set(r["label"] for r in real_dset["train"]))
criterion = nn.CrossEntropyLoss()

for BATCH in BATCH_SIZES:
    print(f"\n=== Running for BATCH_SIZE={BATCH} ===")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # datasets/dataloaders
    pretrain_ds = SPRContrastiveDataset(real_dset["train"])
    pretrain_dl = DataLoader(
        pretrain_ds, batch_size=BATCH, shuffle=True, drop_last=True
    )
    train_ds_sup = SPRSupervisedDataset(real_dset["train"])
    dev_ds_sup = SPRSupervisedDataset(real_dset["dev"])
    train_dl_sup = DataLoader(train_ds_sup, batch_size=BATCH, shuffle=True)
    dev_dl_sup = DataLoader(dev_ds_sup, batch_size=BATCH, shuffle=False)

    # pretrain encoder
    encoder = Encoder(vocab_size).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for ep in range(1, PRE_EPOCHS + 1):
        encoder.train()
        tot = 0
        n = 0
        for batch in pretrain_dl:
            v1, v2 = batch["view1"].to(device), batch["view2"].to(device)
            loss = simclr_loss(encoder(v1), encoder(v2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f"PreEpoch {ep}: loss={tot/n:.4f}")

    # fine-tune
    clf = Classifier(encoder, NUM_CLASSES).to(device)
    ft_opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    batch_metrics = {
        "metrics": {"train_SCWA": [], "val_SCWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, FT_EPOCHS + 1):
        clf.train()
        tot = 0
        n = 0
        for batch in train_dl_sup:
            ids = batch["ids"].to(device)
            y = batch["label"].to(device)
            loss = criterion(clf(ids), y)
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()
            tot += loss.item()
            n += 1
        train_loss = tot / n
        val_loss, val_scwa, preds, gt, seqs = evaluate(clf, dev_dl_sup, criterion)
        ts = datetime.datetime.now().isoformat()
        batch_metrics["losses"]["train"].append(train_loss)
        batch_metrics["losses"]["val"].append(val_loss)
        batch_metrics["metrics"]["train_SCWA"].append(None)
        batch_metrics["metrics"]["val_SCWA"].append(val_scwa)
        batch_metrics["predictions"].append(preds)
        batch_metrics["ground_truth"].append(gt)
        batch_metrics["timestamps"].append(ts)
        print(f"Epoch {ep}: val_loss={val_loss:.4f} | SCWA={val_scwa:.4f}")
    experiment_data["BATCH_SIZE"]["SPR_BENCH"][BATCH] = batch_metrics
    torch.cuda.empty_cache()

# -------------- save --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all results to working/experiment_data.npy")
