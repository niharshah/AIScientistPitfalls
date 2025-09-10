import os, random, time, pathlib, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from collections import Counter

# ----- working dir ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- device --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helpers for metrics -----------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- dataset loading ----------------------
def resolve_spr_path() -> pathlib.Path:
    for p in [
        os.environ.get("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.cwd().parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if p and (pathlib.Path(p) / "train.csv").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("SPR_BENCH csvs not found")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr_path = resolve_spr_path()
spr = load_spr_bench(spr_path)
print({k: len(v) for k, v in spr.items()})


# ---------- vocab & encode -----------------------
def tokenize(s):
    return s.strip().split()


vocab_counter = Counter(
    tok for seq in spr["train"]["sequence"] for tok in tokenize(seq)
)
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = 0, 1

all_labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(all_labels)}
itos_l = {i: l for l, i in ltoi.items()}


def encode_toklist(toks):
    return [stoi.get(t, unk_idx) for t in toks]


def encode_sequence(seq):
    return encode_toklist(tokenize(seq))


# ---------- augmentation -------------------------
def augment_tokens(toks):
    # token masking
    toks2 = [t for t in toks if random.random() > 0.15]
    if not toks2:
        toks2 = toks[:]  # avoid empty
    # local shuffle  (swap two random positions with small prob)
    if len(toks2) > 2 and random.random() < 0.3:
        i, j = random.sample(range(len(toks2)), 2)
        toks2[i], toks2[j] = toks2[j], toks2[i]
    return toks2


# ---------- Datasets -----------------------------
class SPRContrastive(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def collate_contrastive(batch):
    aug1 = [encode_toklist(augment_tokens(tokenize(s))) for s in batch]
    aug2 = [encode_toklist(augment_tokens(tokenize(s))) for s in batch]
    combined = aug1 + aug2
    lengths = [len(s) for s in combined]
    maxlen = max(lengths)
    x = torch.full((len(combined), maxlen), pad_idx, dtype=torch.long)
    for i, seq in enumerate(combined):
        x[i, : len(seq)] = torch.tensor(seq)
    return x.to(device)


class SPRSupervised(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_supervised(batch):
    lengths = [len(b["input"]) for b in batch]
    maxlen = max(lengths)
    x = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["input"])] = b["input"]
    y = torch.stack([b["label"] for b in batch])
    return {"input": x.to(device), "label": y.to(device)}


# ---------- model --------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        mask = (x != pad_idx).unsqueeze(-1)
        z = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.relu(self.proj(z))


class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.proj.out_features, num_labels)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------- loss utils ---------------------------
def nt_xent_loss(z, temperature=0.5):
    z = F.normalize(z, dim=1)
    N = z.size(0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(N, device=device).bool()
    sim.masked_fill_(mask, -9e15)
    B = N // 2
    pos_idx = torch.arange(N, device=device)
    pos_idx = torch.where(pos_idx < B, pos_idx + B, pos_idx - B)
    loss = F.cross_entropy(sim, pos_idx)
    return loss


# ---------- experiment container -----------------
experiment_data = {
    "SPR_contrastive": {
        "metrics": {"val_SWA": [], "val_CWA": [], "val_SCWA": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------- pre-training -------------------------
emb_dim = 128
encoder = Encoder(len(vocab), emb_dim).to(device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
pretrain_loader = DataLoader(
    SPRContrastive(spr["train"]["sequence"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_contrastive,
)

pretrain_epochs = 1
for epoch in range(1, pretrain_epochs + 1):
    encoder.train()
    running = 0.0
    for xb in pretrain_loader:
        optimizer.zero_grad()
        z = encoder(xb)
        loss = nt_xent_loss(z)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    epoch_loss = running / len(pretrain_loader.dataset)
    experiment_data["SPR_contrastive"]["losses"]["pretrain"].append(epoch_loss)
    print(f"Pretrain epoch {epoch}: loss={epoch_loss:.4f}")

# ---------- fine-tune classifier -----------------
clf = Classifier(encoder, len(all_labels)).to(device)
opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    SPRSupervised(spr["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_supervised,
)
val_loader = DataLoader(
    SPRSupervised(spr["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_supervised,
)

best_scwa, best_preds = -1, None
fine_epochs = 3
for epoch in range(1, fine_epochs + 1):
    # --- train ---
    clf.train()
    tr_loss = 0.0
    for batch in train_loader:
        opt_clf.zero_grad()
        logits = clf(batch["input"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        opt_clf.step()
        tr_loss += loss.item() * batch["label"].size(0)
    tr_loss /= len(train_loader.dataset)
    experiment_data["SPR_contrastive"]["losses"]["train"].append(tr_loss)

    # --- val ---
    clf.eval()
    val_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = clf(batch["input"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["label"].size(0)
            preds += logits.argmax(1).cpu().tolist()
            trues += batch["label"].cpu().tolist()
    val_loss /= len(val_loader.dataset)
    experiment_data["SPR_contrastive"]["losses"]["val"].append(val_loss)

    # --- metrics ---
    swa = shape_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    cwa = color_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    sc = scwa(spr["dev"]["sequence"], trues, preds)
    experiment_data["SPR_contrastive"]["metrics"]["val_SWA"].append(swa)
    experiment_data["SPR_contrastive"]["metrics"]["val_CWA"].append(cwa)
    experiment_data["SPR_contrastive"]["metrics"]["val_SCWA"].append(sc)
    experiment_data["SPR_contrastive"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} SCWA={sc:.4f}"
    )

    if sc > best_scwa:
        best_scwa = sc
        best_preds = preds
        best_trues = trues

experiment_data["SPR_contrastive"]["predictions"] = best_preds
experiment_data["SPR_contrastive"]["ground_truth"] = best_trues

# ---------- save --------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
