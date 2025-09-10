import os, time, random, pathlib, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metric helpers -------
def count_shape_variety(seq):  # first char of each token
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_color_variety(seq):  # second char of each token
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1e-9)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1e-9)


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1e-9)


# ---------- dataset utils --------
def locate_spr():
    guesses = [
        os.environ.get("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.cwd().parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for g in guesses:
        g = pathlib.Path(g)
        if g.exists() and (g / "train.csv").exists():
            return g
    raise FileNotFoundError("SPR_BENCH folder not found")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


spr = load_spr(locate_spr())
print({k: len(v) for k, v in spr.items()})


# ---------- vocabulary -----------
def tok(seq):
    return seq.strip().split()


from collections import Counter

counter = Counter(t for seq in spr["train"]["sequence"] for t in tok(seq))
vocab = ["<PAD>", "<UNK>"] + sorted(counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = 0, 1

labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(labels)}
itos_l = {i: l for l, i in ltoi.items()}


def encode_seq(s):
    return [stoi.get(t, unk_idx) for t in tok(s)]


# ---------- augmentations --------
def augment(tokens):
    # 15% token drop
    out = [t for t in tokens if random.random() > 0.15] or tokens
    # swap two tokens with 30% prob
    if len(out) > 2 and random.random() < 0.3:
        i, j = random.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return out


# ---------- datasets -------------
class ContrastiveSPR(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def collate_contrastive(batch):
    views = []
    for s in batch:
        t = tok(s)
        views.append(encode_seq(" ".join(augment(t))))
        views.append(encode_seq(" ".join(augment(t))))
    maxlen = max(map(len, views))
    x = torch.full((len(views), maxlen), pad_idx, dtype=torch.long)
    for i, seq in enumerate(views):
        x[i, : len(seq)] = torch.tensor(seq)
    return x.to(device)


class SupervisedSPR(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return encode_seq(self.seqs[idx]), self.labels[idx]


def collate_supervised(batch):
    xs, ys = zip(*batch)
    maxlen = max(map(len, xs))
    x = torch.full((len(xs), maxlen), pad_idx, dtype=torch.long)
    for i, seq in enumerate(xs):
        x[i, : len(seq)] = torch.tensor(seq)
    return {
        "input": x.to(device),
        "label": torch.tensor(ys, dtype=torch.long).to(device),
    }


# ---------- model ----------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(512, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.d_model = d_model

    def forward(self, x):
        # x: (B, L)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = h.transpose(0, 1)  # transformer expects (L,B,E)
        mask = x == pad_idx
        out = self.encoder(h, src_key_padding_mask=mask).transpose(0, 1)  # (B,L,E)
        # pool (mean over non-pad)
        mask_inv = (~mask).unsqueeze(-1)
        z = (out * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)
        return z  # (B,E)


class ContrastiveModel(nn.Module):
    def __init__(self, vocab_size, num_labels, dim=128):
        super().__init__()
        self.enc = TransformerEncoder(vocab_size, d_model=dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.clf = nn.Linear(dim, num_labels)

    def forward_enc(self, x):
        return self.enc(x)

    def forward_proj(self, x):
        return F.normalize(self.proj(self.enc(x)), dim=1)

    def forward_cls(self, x):
        return self.clf(self.enc(x))


def nt_xent(z, T=0.5):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / T
    N = z.size(0)
    mask = torch.eye(N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    pos = torch.arange(N, device=z.device)
    pos = torch.where(pos < N // 2, pos + N // 2, pos - N // 2)
    return F.cross_entropy(sim, pos)


# ---------- logging structure ----
experiment_data = {
    "SPR": {
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------- pre-training ----------
model = ContrastiveModel(len(vocab), len(labels), dim=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
pre_loader = DataLoader(
    ContrastiveSPR(spr["train"]["sequence"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_contrastive,
)
pre_epochs = 3
for ep in range(1, pre_epochs + 1):
    model.train()
    tot = 0
    for xb in pre_loader:
        opt.zero_grad()
        loss = nt_xent(model.forward_proj(xb))
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)
    epoch_loss = tot / len(pre_loader.dataset)
    experiment_data["SPR"]["losses"]["pretrain"].append(epoch_loss)
    print(f"Pretrain epoch {ep}: loss={epoch_loss:.4f}")

# ---------- fine-tuning ----------
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
criterion = nn.CrossEntropyLoss()
best_scwa, best_preds, best_trues = -1, None, None
opt_ft = torch.optim.Adam(model.parameters(), lr=1e-3)

ft_epochs = 5
for ep in range(1, ft_epochs + 1):
    # training
    model.train()
    tot = 0
    for batch in train_loader:
        opt_ft.zero_grad()
        logits = model.forward_cls(batch["input"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        opt_ft.step()
        tot += loss.item() * batch["label"].size(0)
    tr_loss = tot / len(train_loader.dataset)
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    # validation
    model.eval()
    v_tot = 0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in val_loader:
            logits = model.forward_cls(batch["input"])
            loss = criterion(logits, batch["label"])
            v_tot += loss.item() * batch["label"].size(0)
            preds += logits.argmax(1).cpu().tolist()
            trues += batch["label"].cpu().tolist()
    v_loss = v_tot / len(val_loader.dataset)
    experiment_data["SPR"]["losses"]["val"].append(v_loss)
    swa = shape_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    cwa = color_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    sc = scwa(spr["dev"]["sequence"], trues, preds)
    experiment_data["SPR"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR"]["metrics"]["SCWA"].append(sc)
    experiment_data["SPR"]["timestamps"].append(time.time())
    print(
        f"Epoch {ep}: validation_loss = {v_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} SCWA={sc:.4f}"
    )
    if sc > best_scwa:
        best_scwa, best_preds, best_trues = sc, preds, trues

experiment_data["SPR"]["predictions"] = best_preds
experiment_data["SPR"]["ground_truth"] = best_trues
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Best SCWA:", best_scwa)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
