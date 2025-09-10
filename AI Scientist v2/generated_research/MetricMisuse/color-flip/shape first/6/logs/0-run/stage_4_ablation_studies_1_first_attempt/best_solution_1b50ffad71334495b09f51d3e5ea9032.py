import os, random, pathlib, csv, time
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------
# working directory / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------------------------------------
# metric helpers
def _count_shape(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def _count_color(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def scaa(seqs, y_true, y_pred):
    w = [_count_shape(s) + _count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# -------------------------------------------------------------
# load real or synthetic SPR data
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def _load_csv(fp):
    with open(fp) as f:
        rdr = csv.DictReader(f)
        return [{"sequence": r["sequence"], "label": int(r["label"])} for r in rdr]


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
    print("Real SPR_BENCH not found – generating synthetic data")
    dataset["train"] = _generate_synth(4000)
    dataset["dev"] = _generate_synth(1000)
    dataset["test"] = _generate_synth(1000)
print({k: len(v) for k, v in dataset.items()})

# -------------------------------------------------------------
# vocabulary & encoder helper
PAD, CLS = "<PAD>", "<CLS>"
vocab = {PAD, CLS}
for split in dataset.values():
    for r in split:
        vocab.update(r["sequence"].split())
itos = list(vocab)
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)


def encode(seq, max_len=20):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()]
    ids = ids[:max_len] + [stoi[PAD]] * (max_len - len(ids))
    return ids


# -------------------------------------------------------------
# Dataset for supervised training
class LabelledSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(encode(r["sequence"], self.max_len)),
            torch.tensor(r["label"]),
            r["sequence"],
        )


# -------------------------------------------------------------
# Model definition
class Encoder(nn.Module):
    def __init__(self, vocab, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=stoi[PAD])
        self.bigru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], 1)
        return self.proj(h)


class SPRClassifier(nn.Module):
    def __init__(self, enc, n_cls):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.proj.out_features, n_cls)

    def forward(self, x):
        feat = self.enc(x)
        return self.head(feat), feat


# -------------------------------------------------------------
# training configuration
BATCH = 128
EPOCH_SUP = 4  # purely supervised epochs
MAX_LEN = 20
num_classes = len({r["label"] for r in dataset["train"]})

enc = Encoder(vocab_size, 256).to(device)
model = SPRClassifier(enc, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    LabelledSPR(dataset["train"], MAX_LEN), batch_size=BATCH, shuffle=True
)
dev_loader = DataLoader(LabelledSPR(dataset["dev"], MAX_LEN), batch_size=BATCH)

# -------------------------------------------------------------
# experiment data dict
experiment_data = {
    "no_contrastive": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# -------------------------------------------------------------
# supervised training loop
for ep in range(1, EPOCH_SUP + 1):
    # ---------- train ----------
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

    # ---------- validate ----------
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
        f"Epoch {ep}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f} | SWA={SWA:.3f} CWA={CWA:.3f} SCAA={SCAA:.3f}"
    )

    # ---------- log ----------
    experiment_data["no_contrastive"]["SPR"]["metrics"]["train"].append(
        {"SWA": None, "CWA": None, "SCAA": None}
    )
    experiment_data["no_contrastive"]["SPR"]["metrics"]["val"].append(
        {"SWA": SWA, "CWA": CWA, "SCAA": SCAA}
    )
    experiment_data["no_contrastive"]["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["no_contrastive"]["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["no_contrastive"]["SPR"]["epochs"].append(ep)
    experiment_data["no_contrastive"]["SPR"]["predictions"] = preds
    experiment_data["no_contrastive"]["SPR"]["ground_truth"] = gts

# -------------------------------------------------------------
# save data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
