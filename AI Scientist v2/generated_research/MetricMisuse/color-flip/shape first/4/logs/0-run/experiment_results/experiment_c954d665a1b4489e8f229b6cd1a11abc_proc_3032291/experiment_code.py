import os, random, time, math, pathlib, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ------------------------------------------------------------------
# working directory & experiment data container
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "CoWA": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------------------------------------------------------
# device & seeds
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------
def _shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def _color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [_shape_variety(s) * _color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


# ------------------------------------------------------------------
# Load SPR_BENCH or create synthetic fallback
# ------------------------------------------------------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


root = pathlib.Path("SPR_BENCH")
if root.exists():
    spr = load_spr(root)
else:
    # -------------------------- synthetic data --------------------------
    def make_rows(n, start_id=0):
        shapes, colors = "ABCD", "abcd"
        rows = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            L = random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            rows["id"].append(start_id + i)
            rows["sequence"].append(seq)
            rows["label"].append(int(L % 2))
        return rows

    spr = DatasetDict(
        {
            "train": HFDataset.from_dict(make_rows(1000)),
            "dev": HFDataset.from_dict(make_rows(200, 2000)),
            "test": HFDataset.from_dict(make_rows(200, 4000)),
        }
    )

print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# Vocabulary & encoding
# ------------------------------------------------------------------
all_tokens = set(tok for s in spr["train"]["sequence"] for tok in s.split())
special = ["<PAD>", "<UNK>", "<CLS>"]
tok2idx = {t: i for i, t in enumerate(special + sorted(all_tokens))}
idx2tok = {i: t for t, i in tok2idx.items()}
PAD, UNK, CLS = tok2idx["<PAD>"], tok2idx["<UNK>"], tok2idx["<CLS>"]
vocab_size = len(tok2idx)


def encode(seq: str):
    return [CLS] + [tok2idx.get(tok, UNK) for tok in seq.split()]


# ------------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------------
class SPRSet(Dataset):
    def __init__(self, split, labelled=True):
        self.seq = split["sequence"]
        self.labelled = labelled
        self.labels = split["label"] if labelled else [0] * len(self.seq)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seq[idx],
        }


def collate(batch):
    xs = [b["x"] for b in batch]
    lens = [len(x) for x in xs]
    mx = max(lens)
    padded = torch.full((len(xs), mx), PAD, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : len(x)] = x
    out = {
        "x": padded.to(device),
        "len": torch.tensor(lens, dtype=torch.long).to(device),
        "y": torch.stack([b["y"] for b in batch]).to(device),
        "raw_seq": [b["raw_seq"] for b in batch],
    }
    return out


# ------------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------------
unlabelled = torch.utils.data.ConcatDataset(
    [SPRSet(spr["train"], False), SPRSet(spr["dev"], False), SPRSet(spr["test"], False)]
)
pre_loader = DataLoader(unlabelled, batch_size=128, shuffle=True, collate_fn=collate)
train_loader = DataLoader(
    SPRSet(spr["train"]), batch_size=64, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRSet(spr["dev"]), batch_size=64, shuffle=False, collate_fn=collate
)

n_classes = len(set(spr["train"]["label"]))
print("Classes:", n_classes, "Vocab:", vocab_size)


# ------------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SPRModel(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=2, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 256, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.clf = nn.Linear(d_model, n_classes)

    def forward(self, x, lengths):
        mask = x == PAD
        h = self.pos(self.emb(x))
        h = self.encoder(h, src_key_padding_mask=mask)
        cls = h[:, 0]
        return cls, self.clf(cls)


model = SPRModel(vocab_size, n_classes=n_classes).to(device)


# ------------------------------------------------------------------
# Contrastive pre-training (SimCLR style)
# ------------------------------------------------------------------
def augment(seq_ids, pad_id=PAD):
    # token dropout & local swap
    seq_ids = seq_ids.copy()
    for i in range(1, len(seq_ids)):
        if seq_ids[i] == pad_id:
            break
        if random.random() < 0.1:
            seq_ids[i] = UNK
    if len(seq_ids) > 3 and random.random() < 0.3:
        i = random.randint(1, len(seq_ids) - 2)
        j = min(len(seq_ids) - 1, i + random.randint(1, 2))
        seq_ids[i], seq_ids[j] = seq_ids[j], seq_ids[i]
    return seq_ids


contrast_epochs = 2
temp = 0.07
opt_pre = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(1, contrast_epochs + 1):
    model.train()
    tot = 0
    for batch in pre_loader:
        b = batch["x"].cpu()  # work on cpu for augmentation
        views1, views2 = [], []
        for seq in b.tolist():
            views1.append(augment(seq))
            views2.append(augment(seq))

        def pad(v):
            mx = max(len(s) for s in v)
            t = torch.full((len(v), mx), PAD, dtype=torch.long)
            for i, s in enumerate(v):
                t[i, : len(s)] = torch.tensor(s)
            return t.to(device)

        v1, v2 = pad(views1), pad(views2)
        l1 = torch.tensor([len(s) for s in views1], dtype=torch.long, device=device)
        l2 = torch.tensor([len(s) for s in views2], dtype=torch.long, device=device)

        z1, _ = model(v1, l1)
        z2, _ = model(v2, l2)
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        reps = torch.cat([z1, z2], 0)  # (2N,d)
        sim = torch.mm(reps, reps.t()) / temp  # similarity
        # mask self-similarities
        diag_mask = torch.eye(sim.size(0), device=device).bool()
        sim.masked_fill_(diag_mask, -1e9)
        N = reps.size(0)
        pos_idx = (torch.arange(N, device=device) + z1.size(0)) % N
        loss = nn.functional.cross_entropy(sim, pos_idx)
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        tot += loss.item() * b.size(0)
    print(f"[Contrastive] Epoch {ep}: loss={tot/len(pre_loader.dataset):.4f}")

# ------------------------------------------------------------------
# Supervised fine-tuning
# ------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
fine_tune_epochs = 3

for ep in range(1, fine_tune_epochs + 1):
    # ---- train ----
    model.train()
    t_loss = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        _, logits = model(batch["x"], batch["len"])
        loss = criterion(logits, batch["y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * batch["y"].size(0)
    train_loss = t_loss / len(train_loader.dataset)

    # ---- validation ----
    model.eval()
    v_loss, seqs, y_true, y_pred = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            _, logits = model(batch["x"], batch["len"])
            loss = criterion(logits, batch["y"])
            v_loss += loss.item() * batch["y"].size(0)
            preds = torch.argmax(logits, 1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch["y"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    val_loss = v_loss / len(dev_loader.dataset)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    cowa = complexity_weighted_accuracy(seqs, y_true, y_pred)

    # ---- log ----
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["CoWA"].append(cowa)
    experiment_data["SPR_BENCH"]["predictions"].append(y_pred)
    experiment_data["SPR_BENCH"]["ground_truth"].append(y_true)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CoWA={cowa:.3f}"
    )

# ------------------------------------------------------------------
# Save experiment data
# ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
