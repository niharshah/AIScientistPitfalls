# ──────────────────────────────────────────────────────────────────────────────
#  Context-aware contrastive baseline for SPR ‑- robust to missing data path
# ──────────────────────────────────────────────────────────────────────────────
import os, random, pathlib, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# -----------------------------------------------------------------------------
#  House-keeping
# -----------------------------------------------------------------------------
disable_caching()
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------
#  Locate SPR_BENCH dataset folder
# -----------------------------------------------------------------------------
def locate_spr_bench() -> pathlib.Path:
    """Return a valid pathlib.Path to SPR_BENCH or raise FileNotFoundError."""
    candidates = [
        os.environ.get("SPR_PATH", None),  # user-supplied env var
        "./SPR_BENCH",  # current dir
        "../SPR_BENCH",  # parent dir
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",  # example absolute path
    ]
    for c in candidates:
        if c and os.path.exists(c):
            print(f"Found SPR_BENCH at: {c}")
            return pathlib.Path(c)
    raise FileNotFoundError(
        "SPR_BENCH dataset not found. Set SPR_PATH env-var or place the folder "
        "next to this script."
    )


# -----------------------------------------------------------------------------
#  Data-loading helpers (taken from given SPR.py)
# -----------------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in sequence.strip().split() if tok})


def count_color_variety(sequence: str) -> int:
    return len({tok[1] for tok in sequence.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seq, y_true, y_pred):
    w = [count_shape_variety(s) for s in seq]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1e-9)


def color_weighted_accuracy(seq, y_true, y_pred):
    w = [count_color_variety(s) for s in seq]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1e-9)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa + 1e-9)


# -----------------------------------------------------------------------------
#  Simple tokenizer / vocab
# -----------------------------------------------------------------------------
PAD, UNK = "<PAD>", "<UNK>"


class Vocab:
    def __init__(self):
        self.tok2idx = {PAD: 0, UNK: 1}
        self.idx2tok = [PAD, UNK]

    def add(self, tok):
        if tok not in self.tok2idx:
            self.tok2idx[tok] = len(self.idx2tok)
            self.idx2tok.append(tok)

    def encode(self, toks):  # list[str] -> list[int]
        return [self.tok2idx.get(t, 1) for t in toks]

    def __len__(self):
        return len(self.idx2tok)


class SPRDataset(Dataset):
    def __init__(self, hf_split, vocab: Vocab, build_vocab=False):
        self.text = hf_split["sequence"]
        self.labels = [int(x) for x in hf_split["label"]]
        self.vocab = vocab
        if build_vocab:
            for s in self.text:
                for tok in s.strip().split():
                    self.vocab.add(tok)
        self.encoded = [self.vocab.encode(s.strip().split()) for s in self.text]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encoded[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq": self.text[idx],
        }


def collate(batch):
    maxlen = max(len(x["input_ids"]) for x in batch)
    ids = [
        torch.cat(
            [
                b["input_ids"],
                torch.zeros(maxlen - len(b["input_ids"]), dtype=torch.long),
            ]
        )
        for b in batch
    ]
    labels = torch.stack([b["label"] for b in batch])
    seqs = [b["seq"] for b in batch]
    return {"input_ids": torch.stack(ids), "label": labels, "seq": seqs}


# -----------------------------------------------------------------------------
#  Data augmentations for contrastive learning
# -----------------------------------------------------------------------------
def random_mask(tokens, p=0.3):
    return [t for t in tokens if random.random() > p] or tokens


def make_views(tokens):
    return random_mask(tokens), random_mask(tokens)


# -----------------------------------------------------------------------------
#  Model (biGRU encoder + linear head)
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=64, h=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, h, bidirectional=True, batch_first=True)
        self.h = h

    def forward(self, ids):
        emb = self.embedding(ids)
        out, _ = self.gru(emb)
        return out.mean(1)  # (B, 2h)


class Classifier(nn.Module):
    def __init__(self, enc, num_cls):
        super().__init__()
        self.encoder = enc
        self.head = nn.Linear(enc.h * 2, num_cls)

    def forward(self, ids):
        rep = self.encoder(ids)
        return self.head(rep)


# -----------------------------------------------------------------------------
#  Contrastive loss (SimCLR)
# -----------------------------------------------------------------------------
def simclr_loss(r1, r2, temp=0.5):
    r1, r2 = nn.functional.normalize(r1, dim=1), nn.functional.normalize(r2, dim=1)
    logits = (r1 @ r2.T) / temp
    labels = torch.arange(r1.size(0), device=r1.device)
    return (
        nn.functional.cross_entropy(logits, labels)
        + nn.functional.cross_entropy(logits.T, labels)
    ) / 2


# -----------------------------------------------------------------------------
#  Load data
# -----------------------------------------------------------------------------
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

vocab = Vocab()
train_ds = SPRDataset(spr["train"], vocab, build_vocab=True)
dev_ds = SPRDataset(spr["dev"], vocab)
test_ds = SPRDataset(spr["test"], vocab)

train_loader_contrast = DataLoader(
    train_ds, batch_size=256, shuffle=True, collate_fn=collate
)
train_loader_sup = DataLoader(
    train_ds, batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

num_classes = len(set(train_ds.labels))
print("Vocab:", len(vocab), "| Classes:", num_classes)

# -----------------------------------------------------------------------------
#  Experiment bookkeeping
# -----------------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"contrastive": [], "supervised": {"train": [], "val": []}},
        "predictions": [],
        "ground_truth": [],
    }
}

# -----------------------------------------------------------------------------
#  Contrastive pre-training
# -----------------------------------------------------------------------------
encoder = Encoder(len(vocab)).to(device)
optim_c = torch.optim.Adam(encoder.parameters(), lr=1e-3)

for epoch in range(1, 4):
    encoder.train()
    epoch_loss = 0
    for batch in train_loader_contrast:
        # build two augmented views
        seqs = [s.split() for s in batch["seq"]]
        v1, v2 = [], []
        for toks in seqs:
            a, b = make_views(toks)
            v1.append(torch.tensor(vocab.encode(a), dtype=torch.long))
            v2.append(torch.tensor(vocab.encode(b), dtype=torch.long))
        maxlen = max(max(len(t) for t in v1), max(len(t) for t in v2))
        pad = lambda L: torch.stack(
            [torch.cat([t, torch.zeros(maxlen - len(t), dtype=torch.long)]) for t in L]
        )
        ids1, ids2 = pad(v1).to(device), pad(v2).to(device)
        r1, r2 = encoder(ids1), encoder(ids2)
        loss = simclr_loss(r1, r2)
        optim_c.zero_grad()
        loss.backward()
        optim_c.step()
        epoch_loss += loss.item() * ids1.size(0)
    epoch_loss /= len(train_ds)
    experiment_data["SPR_BENCH"]["losses"]["contrastive"].append(epoch_loss)
    print(f"[Contrast] epoch {epoch} loss {epoch_loss:.4f}")

# -----------------------------------------------------------------------------
#  Supervised fine-tuning
# -----------------------------------------------------------------------------
model = Classifier(encoder, num_classes).to(device)
optim_s = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 6):
    # ---- train ----
    model.train()
    running_loss, preds, gts, seqs = 0, [], [], []
    for batch in train_loader_sup:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch_t["input_ids"])
        loss = criterion(logits, batch_t["label"])
        optim_s.zero_grad()
        loss.backward()
        optim_s.step()
        running_loss += loss.item() * batch_t["input_ids"].size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(batch_t["label"].cpu().tolist())
        seqs.extend(batch["seq"])
    running_loss /= len(train_ds)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    hwa = harmonic_weighted_accuracy(swa, cwa)
    experiment_data["SPR_BENCH"]["losses"]["supervised"]["train"].append(running_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"swa": swa, "cwa": cwa, "hwa": hwa}
    )

    # ---- validation ----
    model.eval()
    val_loss, vp, vg, vs = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch_t["input_ids"])
            loss = criterion(logits, batch_t["label"])
            val_loss += loss.item() * batch_t["input_ids"].size(0)
            vp.extend(logits.argmax(1).cpu().tolist())
            vg.extend(batch_t["label"].cpu().tolist())
            vs.extend(batch["seq"])
    val_loss /= len(dev_ds)
    swa = shape_weighted_accuracy(vs, vg, vp)
    cwa = color_weighted_accuracy(vs, vg, vp)
    hwa = harmonic_weighted_accuracy(swa, cwa)
    experiment_data["SPR_BENCH"]["losses"]["supervised"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"swa": swa, "cwa": cwa, "hwa": hwa}
    )
    experiment_data["SPR_BENCH"]["predictions"] = vp
    experiment_data["SPR_BENCH"]["ground_truth"] = vg
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
    )

# -----------------------------------------------------------------------------
#  Save experiment data
# -----------------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
