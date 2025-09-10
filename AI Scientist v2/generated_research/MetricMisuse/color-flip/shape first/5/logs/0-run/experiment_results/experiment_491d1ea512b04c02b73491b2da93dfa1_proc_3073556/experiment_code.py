import os, random, csv, math, pathlib, time, warnings
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------- Working dir & device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------- Experiment dict ---------------------------------
experiment_data = {
    "SPR_TransformerContrastive": {
        "metrics": {"train": [], "val": []},  # list of dicts per epoch
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------------------- Metric helpers ----------------------------------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def complexity_weight(seq: str) -> int:
    return count_shape_variety(seq) + count_color_variety(seq)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [complexity_weight(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------------------------- Dataset loading ---------------------------------
def generate_synthetic(path: pathlib.Path):
    shapes, colors = list("ABCD"), list("123")

    def gen_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(5, 10))
        )

    def make_csv(name, n):
        with open(path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                seq = gen_seq()
                label = int(count_shape_variety(seq) % 2 == 0)
                w.writerow([i, seq, label])

    make_csv("train.csv", 2000)
    make_csv("dev.csv", 500)
    make_csv("test.csv", 500)


def load_csv_dataset(folder: pathlib.Path) -> Dict[str, List[Dict]]:
    out = {}
    for split in ["train", "dev", "test"]:
        with open(folder / f"{split}.csv") as f:
            rdr = csv.DictReader(f)
            rows = [r for r in rdr]
            for r in rows:
                r["label"] = int(r["label"])
            out[split] = rows
    return out


DATA_PATH = pathlib.Path(os.environ.get("SPR_PATH", "./SPR_BENCH"))
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    generate_synthetic(DATA_PATH)
data = load_csv_dataset(DATA_PATH)
print({k: len(v) for k, v in data.items()})

# ---------------------------- Vocabulary & encoding ---------------------------
PAD, MASK = "<PAD>", "<MASK>"


def build_vocab(samples):
    vocab = {PAD: 0, MASK: 1}
    idx = 2
    for s in samples:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab([r["sequence"] for r in data["train"]])
vocab_size = len(vocab)


def encode(seq: str) -> List[int]:
    return [vocab[t] for t in seq.split()]


# ---------------------------- Dataset class -----------------------------------
class SPRDataset(Dataset):
    def __init__(self, rows, supervised=True):
        self.rows, self.sup = rows, supervised

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        item = {"input": encode(row["sequence"]), "seq": row["sequence"]}
        if self.sup:
            item["label"] = row["label"]
        return item


def collate(batch):
    maxlen = max(len(b["input"]) for b in batch)
    pad_inp = [b["input"] + [0] * (maxlen - len(b["input"])) for b in batch]
    out = {"input": torch.tensor(pad_inp, dtype=torch.long)}
    out["seq"] = [b["seq"] for b in batch]
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


# ---------------------------- Augmentations -----------------------------------
def shuffle_within_window(ids, w=3):
    if len(ids) <= 1:
        return ids
    ids = ids.copy()
    for i in range(0, len(ids), w):
        seg = ids[i : i + w]
        random.shuffle(seg)
        ids[i : i + w] = seg
    return ids


def mask_tokens(ids, p=0.15):
    return [1 if random.random() < p else t for t in ids]


def augment(ids):
    out = shuffle_within_window(ids)
    out = mask_tokens(out)
    return out if out else ids


# ---------------------------- Model -------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_sz, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, nlayers)

    def forward(self, x):
        mask = x == 0  # padding mask
        h = self.enc(self.pos(self.embed(x)), src_key_padding_mask=mask)
        return h.mean(1)  # simple average pooling


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, in_dim=128, n_cls=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_cls)

    def forward(self, x):
        return self.fc(x)


# ---------------------------- Losses ------------------------------------------
def nt_xent(emb, temperature=0.1):
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t() / temperature
    B = emb.size(0) // 2
    labels = torch.arange(B, device=emb.device)
    loss = F.cross_entropy(
        torch.cat([sim[:B, B:], sim[B:, :B]], 0), torch.cat([labels, labels], 0)
    )
    return loss


# ---------------------------- DataLoaders -------------------------------------
batch_size = 128
unsup_loader = DataLoader(
    SPRDataset(data["train"], supervised=False),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
train_loader = DataLoader(
    SPRDataset(data["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(data["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(data["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ---------------------------- Pre-training ------------------------------------
enc = TransformerEncoder(vocab_size).to(device)
proj = ProjectionHead().to(device)
opt_pre = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)

epochs_pre = 2
for ep in range(1, epochs_pre + 1):
    enc.train()
    proj.train()
    total = 0
    steps = 0
    for batch in unsup_loader:
        ids = batch["input"]
        v1 = [augment(s.tolist()) for s in ids]
        v2 = [augment(s.tolist()) for s in ids]

        def pad(seqs):
            ml = max(len(s) for s in seqs)
            return torch.tensor(
                [s + [0] * (ml - len(s)) for s in seqs], dtype=torch.long
            )

        inp = torch.cat([pad(v1), pad(v2)], 0).to(device)
        z = proj(enc(inp))
        loss = nt_xent(z)
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        total += loss.item()
        steps += 1
    print(f"[Pretrain] epoch {ep}/{epochs_pre} loss={total/steps:.4f}")

pretrained_state = enc.state_dict()

# ---------------------------- Fine-tuning -------------------------------------
enc = TransformerEncoder(vocab_size).to(device)
enc.load_state_dict(pretrained_state)
clf = Classifier().to(device)
opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val_loss = 1e9
patience = 3
wait = 0
max_epochs = 15
for epoch in range(1, max_epochs + 1):
    # ---- training ----
    enc.train()
    clf.train()
    tot_loss = 0
    steps = 0
    for batch in train_loader:
        x = batch["input"].to(device)
        y = batch["label"].to(device)
        logits = clf(enc(x))
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_loss += loss.item()
        steps += 1
    train_loss = tot_loss / steps

    # ---- validation ----
    enc.eval()
    clf.eval()
    val_loss = 0
    vsteps = 0
    y_true = []
    y_pred = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)
            logits = clf(enc(x))
            loss = criterion(logits, y)
            val_loss += loss.item()
            vsteps += 1
            pred = logits.argmax(1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(batch["label"].tolist())
            seqs.extend(batch["seq"])
    val_loss /= vsteps
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    comp = complexity_weighted_accuracy(seqs, y_true, y_pred)

    # ---- logging ----
    experiment_data["SPR_TransformerContrastive"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_TransformerContrastive"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_TransformerContrastive"]["metrics"]["train"].append(
        {"epoch": epoch}
    )
    experiment_data["SPR_TransformerContrastive"]["metrics"]["val"].append(
        {"epoch": epoch, "SWA": swa, "CWA": cwa, "CompWA": comp}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CompWA={comp:.3f}"
    )

    # ---- early stopping ----
    if val_loss + 1e-6 < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        best_state = (enc.state_dict(), clf.state_dict())
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------------------------- Test evaluation ---------------------------------
enc.load_state_dict(best_state[0])
clf.load_state_dict(best_state[1])
enc.eval()
clf.eval()
y_true = []
y_pred = []
seqs = []
with torch.no_grad():
    for batch in test_loader:
        x = batch["input"].to(device)
        logits = clf(enc(x))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(batch["label"])
        seqs.extend(batch["seq"])
experiment_data["SPR_TransformerContrastive"]["predictions"] = y_pred
experiment_data["SPR_TransformerContrastive"]["ground_truth"] = y_true
test_comp = complexity_weighted_accuracy(seqs, y_true, y_pred)
print(f"TEST Complexity-Weighted Accuracy = {test_comp:.3f}")

# ---------------------------- Save --------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
