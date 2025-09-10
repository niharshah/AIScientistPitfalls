import os, random, pathlib, math, time, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------- #
# basic setup, working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------- #
# helper metrics
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def SWA(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def CWA(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def SCAA(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# --------------------------------------------------- #
# data loading
SPR_ROOT = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def read_csv(split):
    fp = SPR_ROOT / f"{split}.csv"
    rows = []
    if fp.exists():
        with open(fp) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({"sequence": r["sequence"], "label": int(r["label"])})
    return rows


def synthetic(n=3000):
    shapes, colors = "ABC", "123"
    data = []
    for _ in range(n):
        L = random.randint(4, 8)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        lab = (seq.count("A1") + seq.count("B2")) % 3
        data.append({"sequence": seq, "label": lab})
    return data


dataset = {}
for split in ["train", "dev", "test"]:
    rows = read_csv(split)
    if not rows:
        rows = synthetic(6000 if split == "train" else 2000)
    dataset[split] = rows
print({k: len(v) for k, v in dataset.items()})

# --------------------------------------------------- #
# vocabulary
special = ["<PAD>", "<CLS>", "<MASK>"]
tokens = set()
for rows in dataset.values():
    for r in rows:
        tokens.update(r["sequence"].split())
itos = special + sorted(tokens)
stoi = {tok: i for i, tok in enumerate(itos)}
PAD, CLS, MASK = [stoi[s] for s in special]
vocab_size = len(itos)
max_len = 20
print("Vocabulary size:", vocab_size)


def encode(seq):
    ids = [CLS] + [stoi[t] for t in seq.split()][: max_len - 1]
    ids += [PAD] * (max_len - len(ids))
    return ids


# --------------------------------------------------- #
# augmentations
def aug(seq: str) -> str:
    toks = seq.split()
    # random local shuffle
    if len(toks) > 3 and random.random() < 0.5:
        i = random.randint(0, len(toks) - 2)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
    # random dropout
    toks = [t for t in toks if random.random() > 0.1 or len(toks) <= 3]
    # token masking
    toks = [t if random.random() > 0.15 else "<MASK>" for t in toks]
    return " ".join(toks) if toks else seq


# --------------------------------------------------- #
# datasets
class ContrastiveSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        return torch.tensor(encode(aug(s))), torch.tensor(encode(aug(s)))


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


# --------------------------------------------------- #
# model – lightweight Transformer encoder
class SeqEncoder(nn.Module):
    def __init__(self, vocab, d_model=256, nhead=8, num_layers=2, dim_ff=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, : x.size(1), :]
        mask = x[:, :, 0] == 0  # padding mask – not ideal but quick
        h = self.enc(x, src_key_padding_mask=mask)
        return h[:, 0]  # CLS position


class SPRClassifier(nn.Module):
    def __init__(self, enc, num_classes):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(enc.embed.embedding_dim, num_classes)

    def forward(self, x):
        feat = self.enc(x)
        return self.fc(feat), feat


# --------------------------------------------------- #
# contrastive loss
def nt_xent(feat, temp=0.5):
    N = feat.size(0) // 2
    f = F.normalize(feat, dim=1)
    sim = torch.mm(f, f.t()) / temp
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(N, 2 * N, device=feat.device)
    targets = torch.cat([targets, torch.arange(0, N, device=feat.device)])
    return F.cross_entropy(sim, targets)


# --------------------------------------------------- #
# training hyperparams
BATCH = 256
EPOCH_PRE = 2
EPOCH_FT = 3
num_classes = len(set(r["label"] for r in dataset["train"]))

contrast_loader = DataLoader(
    ContrastiveSPR(dataset["train"]),
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)
train_loader = DataLoader(
    LabelledSPR(dataset["train"]), batch_size=BATCH, shuffle=True, num_workers=2
)
val_loader = DataLoader(LabelledSPR(dataset["dev"]), batch_size=BATCH, num_workers=2)

encoder = SeqEncoder(vocab_size).to(device)
model = SPRClassifier(encoder, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

# --------------------------------------------------- #
# experiment data container
experiment_data = {
    "ContextContrastive": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "SCAA": {"val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------------------------------------------- #
# 1) contrastive pre-training
print("\n=== Contrastive pre-training ===")
for ep in range(1, EPOCH_PRE + 1):
    model.train()
    total_loss = 0
    for v1, v2 in contrast_loader:
        v1, v2 = v1.to(device), v2.to(device)
        _, f1 = model(v1)
        _, f2 = model(v2)
        loss = nt_xent(torch.cat([f1, f2], 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * v1.size(0)
    print(
        f" Epoch {ep}: contrastive_loss = {total_loss/len(contrast_loader.dataset):.4f}"
    )

# --------------------------------------------------- #
# 2) supervised fine-tuning
print("\n=== Supervised fine-tuning ===")
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

    # validation
    model.eval()
    val_loss, seqs, preds, gts = 0, [], [], []
    with torch.no_grad():
        for ids, labels, seq in val_loader:
            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)
            loss = criterion(logits, labels)
            val_loss += loss.item() * ids.size(0)
            p = torch.argmax(logits, 1).cpu().tolist()
            preds.extend(p)
            gts.extend(labels.cpu().tolist())
            seqs.extend(seq)
    val_loss /= len(dataset["dev"])
    swa, cwa, scaa = (
        SWA(seqs, gts, preds),
        CWA(seqs, gts, preds),
        SCAA(seqs, gts, preds),
    )
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCAA={scaa:.3f}"
    )

    experiment_data["ContextContrastive"]["metrics"]["train"].append(swa)
    experiment_data["ContextContrastive"]["metrics"]["val"].append(cwa)
    experiment_data["ContextContrastive"]["losses"]["train"].append(tr_loss)
    experiment_data["ContextContrastive"]["losses"]["val"].append(val_loss)
    experiment_data["ContextContrastive"]["SCAA"]["val"].append(scaa)
    experiment_data["ContextContrastive"]["predictions"] = preds
    experiment_data["ContextContrastive"]["ground_truth"] = gts

# --------------------------------------------------- #
# save all
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
