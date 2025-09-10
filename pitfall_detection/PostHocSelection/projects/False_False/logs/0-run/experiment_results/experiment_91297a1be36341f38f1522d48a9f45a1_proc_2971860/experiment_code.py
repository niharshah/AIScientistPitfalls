import os, time, random, pathlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from collections import Counter

# ------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# metric helpers ----------------------------------------------------
def count_shape_variety(seq):  # shape = first char of token
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):  # color = second char of token
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return (
        sum(w0 for w0, t, p in zip(w, y_true, y_pred) if t == p) / sum(w)
        if sum(w)
        else 0.0
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return (
        sum(w0 for w0, t, p in zip(w, y_true, y_pred) if t == p) / sum(w)
        if sum(w)
        else 0.0
    )


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return (
        sum(w0 for w0, t, p in zip(w, y_true, y_pred) if t == p) / sum(w)
        if sum(w)
        else 0.0
    )


# ------------------------------------------------------------------
# SPR dataset loading ----------------------------------------------
def resolve_spr_path() -> pathlib.Path:
    for p in [
        os.getenv("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.cwd().parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if p and (pathlib.Path(p) / "train.csv").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("Could not locate SPR_BENCH")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr_path = resolve_spr_path()
spr = load_spr_bench(spr_path)
print({k: len(v) for k, v in spr.items()})


# ------------------------------------------------------------------
# vocabulary --------------------------------------------------------
def tokenize(s):
    return s.strip().split()


vocab_counter = Counter(tok for s in spr["train"]["sequence"] for tok in tokenize(s))
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = 0, 1


def encode_tokens(toks):
    return [stoi.get(t, unk_idx) for t in toks]


def encode_seq(seq):
    return encode_tokens(tokenize(seq))


labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(labels)}
itos_l = {i: l for l, i in ltoi.items()}


# ------------------------------------------------------------------
# ORIGINAL augmentations (kept for reference but unused in ablation)-
def augment_tokens(toks):
    toks = [t for t in toks if random.random() > 0.15] or toks
    if len(toks) > 3 and random.random() < 0.3:
        i, j = sorted(random.sample(range(len(toks)), 2))
        toks[i:j] = reversed(toks[i:j])
    return toks


MAX_LEN = 128


class ContrastiveSPR(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


# --------- NO-VIEW-AUGMENTATION COLLATE FUNCTION ------------------
def collate_contrastive_noaug(batch):
    views = []
    for s in batch:
        enc = encode_tokens(tokenize(s))  # NO augmentation
        views.append(enc)
        views.append(enc)  # identical pair
    maxlen = min(MAX_LEN, max(len(v) for v in views))
    x = torch.full((len(views), maxlen), pad_idx, dtype=torch.long)
    for i, seq in enumerate(views):
        seq = seq[:maxlen]
        x[i, : len(seq)] = torch.tensor(seq)
    return x.to(device)


# supervised dataset ----------------------------------------------
class SupervisedSPR(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labs = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(
                encode_seq(self.seqs[idx])[:MAX_LEN], dtype=torch.long
            ),
            "label": torch.tensor(self.labs[idx], dtype=torch.long),
        }


def collate_supervised(batch):
    maxlen = max(len(b["input"]) for b in batch)
    x = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["input"])] = b["input"]
    y = torch.stack([b["label"] for b in batch])
    return {"input": x.to(device), "label": y.to(device)}


# ------------------------------------------------------------------
# Transformer encoder ----------------------------------------------
class SPRTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_heads=4, n_layers=2, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_len, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.emb_dim = emb_dim

    def forward(self, x):
        pos_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos_ids)
        mask = x == pad_idx
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_inv = (~mask).unsqueeze(-1)
        return (h * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)


class ProjectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, z):
        return self.fc(z)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder.emb_dim, num_labels)

    def forward(self, x):
        return self.cls(self.encoder(x))


# ------------------------------------------------------------------
def nt_xent(z, temp=0.5):
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim.masked_fill_(eye, -9e15)
    N = z.size(0) // 2
    pos = torch.arange(sim.size(0), device=sim.device)
    pos = torch.where(pos < N, pos + N, pos - N)
    return F.cross_entropy(sim, pos)


# ------------------------------------------------------------------
# experiment data dict ---------------------------------------------
experiment_data = {
    "no_view_aug": {
        "SPR_transformer": {
            "metrics": {"val_SWA": [], "val_CWA": [], "val_SCWA": []},
            "losses": {"pretrain": [], "train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ------------------------------------------------------------------
# Contrastive pre-training (no view augmentation) ------------------
emb_dim = 128
encoder = SPRTransformer(len(vocab), emb_dim=emb_dim).to(device)
proj = ProjectionHead(emb_dim).to(device)
opt_pre = torch.optim.Adam(
    list(encoder.parameters()) + list(proj.parameters()), lr=1e-3
)

pre_loader = DataLoader(
    ContrastiveSPR(spr["train"]["sequence"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_contrastive_noaug,
)

pre_epochs = 2
for ep in range(1, pre_epochs + 1):
    encoder.train()
    proj.train()
    running = 0.0
    for xb in pre_loader:
        opt_pre.zero_grad()
        z = proj(encoder(xb))
        loss = nt_xent(z)
        loss.backward()
        opt_pre.step()
        running += loss.item() * xb.size(0)
    ep_loss = running / len(pre_loader.dataset)
    experiment_data["no_view_aug"]["SPR_transformer"]["losses"]["pretrain"].append(
        ep_loss
    )
    print(f"Pretrain Epoch {ep}: loss={ep_loss:.4f}")

# ------------------------------------------------------------------
# Supervised fine-tuning -------------------------------------------
model = SPRModel(encoder, len(labels)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    SupervisedSPR(spr["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_supervised,
)
val_loader = DataLoader(
    SupervisedSPR(spr["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_supervised,
)

best_scwa = -1
best_preds = []
best_trues = []
fine_epochs = 4
for ep in range(1, fine_epochs + 1):
    # train
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        opt.zero_grad()
        logits = model(batch["input"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        opt.step()
        tr_loss += loss.item() * batch["label"].size(0)
    tr_loss /= len(train_loader.dataset)
    experiment_data["no_view_aug"]["SPR_transformer"]["losses"]["train"].append(tr_loss)
    # val
    model.eval()
    val_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["label"].size(0)
            preds += logits.argmax(1).cpu().tolist()
            trues += batch["label"].cpu().tolist()
    val_loss /= len(val_loader.dataset)
    experiment_data["no_view_aug"]["SPR_transformer"]["losses"]["val"].append(val_loss)
    swa = shape_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    cwa = color_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    sc = scwa(spr["dev"]["sequence"], trues, preds)
    ed = experiment_data["no_view_aug"]["SPR_transformer"]
    ed["metrics"]["val_SWA"].append(swa)
    ed["metrics"]["val_CWA"].append(cwa)
    ed["metrics"]["val_SCWA"].append(sc)
    ed["timestamps"].append(time.time())
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} SCWA={sc:.4f}"
    )
    if sc > best_scwa:
        best_scwa = sc
        best_preds = preds
        best_trues = trues

experiment_data["no_view_aug"]["SPR_transformer"]["predictions"] = best_preds
experiment_data["no_view_aug"]["SPR_transformer"]["ground_truth"] = best_trues

# ------------------------------------------------------------------
# save
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Experiment data saved to working/experiment_data.npy")
