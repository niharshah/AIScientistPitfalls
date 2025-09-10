import os, random, pathlib, csv, time, numpy as np
from typing import List, Dict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# metric helpers
def _count_shape(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def _count_color(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [_count_shape(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [_count_color(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def scaa(seqs, y_t, y_p):
    w = [_count_shape(s) + _count_color(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


# ---------------------------------------------------------------------
# data (real or synthetic)
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def _load_csv(fp):
    with open(fp) as f:
        return [
            {"sequence": r["sequence"], "label": int(r["label"])}
            for r in csv.DictReader(f)
        ]


def _generate_synth(n=3000, max_len=8):
    shapes, colors = list("ABC"), list("123")

    def rule(seq):
        return sum(tok == "A1" for tok in seq) % 2

    rows = []
    for _ in range(n):
        toks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, max_len))
        ]
        rows.append({"sequence": " ".join(toks), "label": rule(toks)})
    return rows


dataset: Dict[str, List[Dict]] = {}
try:
    if SPR_PATH.exists():
        for split in ["train", "dev", "test"]:
            dataset[split] = _load_csv(SPR_PATH / f"{split}.csv")
    else:
        raise FileNotFoundError
except Exception:
    print("Real SPR_BENCH not found – using synthetic data")
    dataset["train"] = _generate_synth(4000)
    dataset["dev"] = _generate_synth(1000)
    dataset["test"] = _generate_synth(1000)
print({k: len(v) for k, v in dataset.items()})

# ---------------------------------------------------------------------
# vocab & encoding
PAD, CLS = "<PAD>", "<CLS>"
vocab = set([PAD, CLS])
for split in dataset.values():
    for r in split:
        vocab.update(r["sequence"].split())
itos = list(vocab)
stoi = {tok: i for i, tok in enumerate(itos)}


def encode(seq, max_len=20):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()]
    ids = ids[:max_len] + [stoi[PAD]] * (max_len - len(ids))
    return ids


vocab_size = len(itos)


# ---------------------------------------------------------------------
# augmentations
def aug(seq: str) -> str:
    toks = seq.split()
    if len(toks) > 1 and random.random() < 0.7:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    toks = [t for t in toks if random.random() > 0.1]
    if not toks:
        toks = ["A1"]
    return " ".join(toks)


# ---------------------------------------------------------------------
# datasets
class ContrastiveSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        return torch.tensor(encode(aug(s), self.max_len)), torch.tensor(
            encode(aug(s), self.max_len)
        )


class LabelledSPR(Dataset):
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


# ---------------------------------------------------------------------
# Encoder (Uni-directional GRU, doubled hidden size)
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.gru = nn.GRU(d_model, d_model * 2, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)  # h shape [1,B,hidden]
        h = h.squeeze(0)  # [B,hidden]
        return self.proj(h)


class SPRClassifier(nn.Module):
    def __init__(self, enc, n_classes):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.proj.out_features, n_classes)

    def forward(self, x):
        feat = self.enc(x)
        return self.head(feat), feat


# ---------------------------------------------------------------------
def nt_xent(feats, temp=0.5):
    f = F.normalize(feats, dim=1)
    N = f.size(0) // 2
    sim = torch.mm(f, f.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(N, 2 * N, device=f.device)
    targets = torch.cat([targets, torch.arange(0, N, device=f.device)])
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------
# config
BATCH = 128
EPOCH_PRE = 2
EPOCH_FT = 2
MAX_LEN = 20
num_classes = len({r["label"] for r in dataset["train"]})

enc = Encoder(vocab_size, 256).to(device)
model = SPRClassifier(enc, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

contrast_loader = DataLoader(
    ContrastiveSPR(dataset["train"], MAX_LEN), batch_size=BATCH, shuffle=True
)
train_loader = DataLoader(
    LabelledSPR(dataset["train"], MAX_LEN), batch_size=BATCH, shuffle=True
)
dev_loader = DataLoader(LabelledSPR(dataset["dev"], MAX_LEN), batch_size=BATCH)

# experiment data container
experiment_data = {
    "UniGRU_no_backward": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ---------------------------------------------------------------------
# contrastive pre-training
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
    print(f"Contrastive epoch {ep}: loss={total/len(dataset['train']):.4f}")

# ---------------------------------------------------------------------
# fine-tuning
criterion = nn.CrossEntropyLoss()
for ep in range(1, EPOCH_FT + 1):
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

    model.eval()
    val_loss = 0
    preds = []
    gts = []
    seqs = []
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
    SWA = shape_weighted_accuracy(seqs, gts, preds)
    CWA = color_weighted_accuracy(seqs, gts, preds)
    SCAA = scaa(seqs, gts, preds)
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | SWA={SWA:.3f} CWA={CWA:.3f} SCAA={SCAA:.3f}"
    )

    # logging
    experiment_data["UniGRU_no_backward"]["SPR"]["metrics"]["train"].append(
        {"SWA": None, "CWA": None, "SCAA": None}
    )
    experiment_data["UniGRU_no_backward"]["SPR"]["metrics"]["val"].append(
        {"SWA": SWA, "CWA": CWA, "SCAA": SCAA}
    )
    experiment_data["UniGRU_no_backward"]["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["UniGRU_no_backward"]["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["UniGRU_no_backward"]["SPR"]["predictions"] = preds
    experiment_data["UniGRU_no_backward"]["SPR"]["ground_truth"] = gts
    experiment_data["UniGRU_no_backward"]["SPR"]["epochs"].append(ep)

# ---------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
