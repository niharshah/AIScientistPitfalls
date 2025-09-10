import os, random, pathlib, csv, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- basic set-up ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- helper metrics --------------
def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in sequence.strip().split() if tok})


def count_color_variety(sequence: str) -> int:
    return len({tok[1] for tok in sequence.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def structural_complexity_adjusted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


# ---------------- data loading ----------------
SPR_ROOT = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def _load_csv(path):
    rows = []
    if path.exists():
        with open(path) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({"sequence": r["sequence"], "label": int(r["label"])})
    return rows


def _toy_split(n):
    shapes, colours = "ABCDEF", "123456"
    data = []
    for _ in range(n):
        length = random.randint(4, 9)
        seq = " ".join(
            random.choice(shapes) + random.choice(colours) for _ in range(length)
        )
        label = (seq.count("A1") + seq.count("B2") + length) % 4
        data.append({"sequence": seq, "label": label})
    return data


dataset = {}
for split, n in [("train", 4000), ("dev", 1000), ("test", 1000)]:
    rows = _load_csv(SPR_ROOT / f"{split}.csv")
    if not rows:
        rows = _toy_split(n)
    dataset[split] = rows
print({k: len(v) for k, v in dataset.items()})

# -------------- vocabulary --------------------
tokens = set(
    tok for rows in dataset.values() for r in rows for tok in r["sequence"].split()
)
PAD, CLS, MASK = "<PAD>", "<CLS>", "<MASK>"
itos = [PAD, CLS, MASK] + sorted(tokens)
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)
max_len = 20


def encode(seq: str):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()][: max_len - 1]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


# ------------- data augmentation --------------
def augment(seq: str):
    toks = seq.split()
    if len(toks) == 0:
        return seq
    # mask 15%
    for i in range(len(toks)):
        if random.random() < 0.15:
            toks[i] = MASK
    # local shuffle
    if len(toks) > 3 and random.random() < 0.5:
        i = random.randint(0, len(toks) - 2)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
    # rotation
    if len(toks) > 1 and random.random() < 0.3:
        k = random.randint(1, len(toks) - 1)
        toks = toks[k:] + toks[:k]
    return " ".join(toks)


# ------------- PyTorch datasets ---------------
class ContrastiveSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        return torch.tensor(encode(augment(s))), torch.tensor(encode(augment(s)))


class LabelledSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(encode(r["sequence"])),
            torch.tensor(r["label"]),
            r["sequence"],
        )


# -------------- model -------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, 0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, nlayers)
        self.d_model = d_model

    def forward(self, x):
        mask = x == 0
        h = self.tok_emb(x) + self.pos_emb[:, : x.size(1)]
        z = self.enc(h, src_key_padding_mask=mask)
        # CLS token representation
        return z[:, 0]


class SPRNet(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(
            encoder.d_model, encoder.d_model
        )  # projection head for contrastive
        self.classifier = nn.Linear(encoder.d_model, num_classes)

    def features(self, x):
        return self.encoder(x)

    def forward(self, x, mode="clf"):
        feat = self.features(x)
        if mode == "contrast":
            return F.normalize(self.proj(feat), dim=1)
        else:
            return self.classifier(feat)


# ------------- loss ---------------------------
def nt_xent(feat, T=0.5):
    feat = F.normalize(feat, dim=1)
    sim = torch.matmul(feat, feat.T) / T
    batch = feat.size(0) // 2
    labels = torch.arange(batch, 2 * batch, device=feat.device)
    labels = torch.cat([labels, torch.arange(0, batch, device=feat.device)])
    sim.fill_diagonal_(-9e15)
    return F.cross_entropy(sim, labels)


# ------------- training params ---------------
BATCH = 256
PRE_EPOCHS, FT_EPOCHS = 2, 3
num_classes = len({r["label"] for r in dataset["train"]})

contrast_loader = DataLoader(
    ContrastiveSPR(dataset["train"]), batch_size=BATCH, shuffle=True, drop_last=True
)
train_loader = DataLoader(LabelledSPR(dataset["train"]), batch_size=BATCH, shuffle=True)
dev_loader = DataLoader(LabelledSPR(dataset["dev"]), batch_size=BATCH)

model = SPRNet(TransformerEncoder(vocab_size), num_classes).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "SCAA": {"val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------- contrastive pre-train ----------
print("--- Contrastive pre-training ---")
for ep in range(1, PRE_EPOCHS + 1):
    model.train()
    total = 0
    for v1, v2 in contrast_loader:
        v1, v2 = v1.to(device), v2.to(device)
        f1 = model(v1, mode="contrast")
        f2 = model(v2, mode="contrast")
        loss = nt_xent(torch.cat([f1, f2], 0))
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * v1.size(0)
    print(f"Pre-Epoch {ep}: contrastive_loss={total/len(contrast_loader.dataset):.4f}")

# ------------- fine-tune ----------------------
criterion = nn.CrossEntropyLoss()
print("--- Supervised fine-tuning ---")
for ep in range(1, FT_EPOCHS + 1):
    # train
    model.train()
    tr_loss = 0
    for ids, labels, _ in train_loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids, mode="clf")
        loss = criterion(logits, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        tr_loss += loss.item() * ids.size(0)
    tr_loss /= len(dataset["train"])
    # validate
    model.eval()
    val_loss, preds, gts, seqs = 0, [], [], []
    with torch.no_grad():
        for ids, labels, s in dev_loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids, mode="clf")
            loss = criterion(logits, labels)
            val_loss += loss.item() * ids.size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(labels.cpu().tolist())
            seqs.extend(s)
    val_loss /= len(dataset["dev"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    scaa = structural_complexity_adjusted_accuracy(seqs, gts, preds)
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCAA={scaa:.3f}"
    )
    # record
    experiment_data["SPR"]["metrics"]["train"].append(swa)
    experiment_data["SPR"]["metrics"]["val"].append(cwa)
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["SCAA"]["val"].append(scaa)
    experiment_data["SPR"]["predictions"] = preds
    experiment_data["SPR"]["ground_truth"] = gts

# ------------- save everything ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
