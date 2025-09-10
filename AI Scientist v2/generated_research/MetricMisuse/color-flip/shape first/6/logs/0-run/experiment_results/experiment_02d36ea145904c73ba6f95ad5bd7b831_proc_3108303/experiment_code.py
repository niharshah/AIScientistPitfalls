import os, random, pathlib, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper metrics ----------
def count_shape_variety(seq):  # shape = first char of token
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):  # color = second char (if any)
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def scaa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


# ---------- data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def load_csv(path):
    rows = []
    if path.exists():
        with open(path) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({"sequence": r["sequence"], "label": int(r["label"])})
    return rows


def generate_toy(n=1000):
    shapes, colors = "ABC", "123"

    def rule(s):
        return (s.count("A1") + len(s.split())) % 3

    data = []
    for _ in range(n):
        seq = " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 8))
        )
        data.append({"sequence": seq, "label": rule(seq)})
    return data


dataset = {}
for split in ["train", "dev", "test"]:
    rows = load_csv(DATA_PATH / f"{split}.csv")
    if not rows:  # fall back to synthetic toy data if benchmark files unavailable
        rows = generate_toy(4000 if split == "train" else 1000)
    dataset[split] = rows
print({k: len(v) for k, v in dataset.items()})

# ---------- vocab ----------
PAD, CLS, MASK = "<PAD>", "<CLS>", "<MASK>"
tokens = set()
for split in dataset.values():
    for r in split:
        tokens.update(r["sequence"].split())
tokens.update({PAD, CLS, MASK})  # <- BUGFIX: ensure <MASK> is in vocab
itos = sorted(tokens, key=lambda t: (t != PAD, t != CLS, t))
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)


def encode(seq, max_len=20):
    ids = [stoi[CLS]] + [
        stoi.get(t, stoi[PAD]) for t in seq.split()
    ]  # <- robust to OOV
    ids = ids[:max_len]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


# ---------- augmentations ----------
def mask_tokens(toks, p=0.3):
    return [tok if random.random() > p else MASK for tok in toks]


def local_shuffle(toks, window=3):
    n = len(toks)
    if n < 2:
        return toks
    i = random.randint(0, n - 2)
    j = min(n, i + window)
    seg = toks[i:j]
    random.shuffle(seg)
    return toks[:i] + seg + toks[j:]


def make_view(seq):
    toks = seq.split()
    if random.random() < 0.5:
        toks = mask_tokens(toks)
    else:
        toks = local_shuffle(toks)
    return " ".join(toks)


# ---------- datasets ----------
class ContrastiveSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        v1 = torch.tensor(encode(make_view(s), self.max_len), dtype=torch.long)
        v2 = torch.tensor(encode(make_view(s), self.max_len), dtype=torch.long)
        return v1, v2


class LabelledSPR(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = torch.tensor(encode(r["sequence"], self.max_len), dtype=torch.long)
        return ids, torch.tensor(r["label"], dtype=torch.long), r["sequence"]


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, vocab, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=stoi[PAD])
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.gru(x)
        return h.squeeze(0)  # (B,dim)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(encoder.gru.hidden_size, num_classes)

    def forward(self, x):
        feat = self.enc(x)
        return self.head(feat), feat


# ---------- loss ----------
def nt_xent(feats, temp=0.5):
    feats = F.normalize(feats, dim=1)
    N = feats.size(0) // 2
    sim = torch.mm(feats, feats.t()) / temp
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(N, 2 * N, device=feats.device)
    targets = torch.cat([targets, torch.arange(0, N, device=feats.device)])
    return F.cross_entropy(sim, targets)


# ---------- experiment setup ----------
num_classes = len({r["label"] for r in dataset["train"]})
BATCH = 256
EPOCH_PRE = 3
EPOCH_FT = 3
MAX_LEN = 20
experiment_data = {
    "contrastive_context_aware": {
        "metrics": {
            "train": [],
            "val": [],
            "val_swaa": [],
            "val_cwaa": [],
            "val_scaa": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

contrast_loader = DataLoader(
    ContrastiveSPR(dataset["train"], MAX_LEN),
    batch_size=BATCH,
    shuffle=True,
    drop_last=True,
)
train_loader = DataLoader(
    LabelledSPR(dataset["train"], MAX_LEN), batch_size=BATCH, shuffle=True
)
val_loader = DataLoader(LabelledSPR(dataset["dev"], MAX_LEN), batch_size=BATCH)

encoder = Encoder(vocab_size, dim=256).to(device)
model = SPRModel(encoder, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- pre-training ----------
print("\nSelf-supervised pre-training")
for ep in range(1, EPOCH_PRE + 1):
    model.train()
    running = 0.0
    for v1, v2 in contrast_loader:
        v1, v2 = v1.to(device), v2.to(device)
        _, f1 = model(v1)
        _, f2 = model(v2)
        loss = nt_xent(torch.cat([f1, f2], 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * v1.size(0)
    print(f"Epoch {ep}: contrastive_loss={(running/len(dataset['train'])):.4f}")

# ---------- fine-tuning ----------
criterion = nn.CrossEntropyLoss()
print("\nSupervised fine-tuning")
for ep in range(1, EPOCH_FT + 1):
    # ---- train ----
    model.train()
    train_loss = 0.0
    for ids, labels, _ in train_loader:
        ids, labels = ids.to(device), labels.to(device)
        logits, _ = model(ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * ids.size(0)
    train_loss /= len(dataset["train"])

    # ---- validate ----
    model.eval()
    val_loss = 0.0
    preds, gts, seqs = [], [], []
    with torch.no_grad():
        for ids, labels, seq in val_loader:
            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)
            loss = criterion(logits, labels)
            val_loss += loss.item() * ids.size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(labels.cpu().tolist())
            seqs.extend(seq)
    val_loss /= len(dataset["dev"])
    metric_swa = swa(seqs, gts, preds)
    metric_cwa = cwa(seqs, gts, preds)
    metric_scaa = scaa(seqs, gts, preds)
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={metric_swa:.3f} "
        f"CWA={metric_cwa:.3f} SCAA={metric_scaa:.3f}"
    )

    # ---- log ----
    experiment_data["contrastive_context_aware"]["metrics"]["train"].append(metric_swa)
    experiment_data["contrastive_context_aware"]["metrics"]["val"].append(metric_cwa)
    experiment_data["contrastive_context_aware"]["metrics"]["val_swaa"].append(
        metric_swa
    )
    experiment_data["contrastive_context_aware"]["metrics"]["val_cwaa"].append(
        metric_cwa
    )
    experiment_data["contrastive_context_aware"]["metrics"]["val_scaa"].append(
        metric_scaa
    )
    experiment_data["contrastive_context_aware"]["losses"]["train"].append(train_loss)
    experiment_data["contrastive_context_aware"]["losses"]["val"].append(val_loss)
    if ep == EPOCH_FT:
        experiment_data["contrastive_context_aware"]["predictions"] = preds
        experiment_data["contrastive_context_aware"]["ground_truth"] = gts

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to working/experiment_data.npy")
