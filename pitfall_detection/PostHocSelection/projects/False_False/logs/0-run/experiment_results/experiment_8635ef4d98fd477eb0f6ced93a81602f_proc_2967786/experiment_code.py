import os, random, time, pathlib, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from collections import Counter

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics helpers -------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [w0 if t == p else 0 for w0, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [w0 if t == p else 0 for w0, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def scwa_metric(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [w0 if t == p else 0 for w0, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- load SPR_BENCH ---------
def find_spr():
    for p in [
        os.environ.get("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if p and (pathlib.Path(p) / "train.csv").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("SPR_BENCH not found")


root = find_spr()


def load_spr_bench(path):
    def _ld(x):
        return load_dataset(
            "csv", data_files=str(path / x), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ---------- vocabulary -------------
def tok(seq):
    return seq.strip().split()


vocab_counter = Counter(t for s in spr["train"]["sequence"] for t in tok(s))
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = 0, 1
lbls = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(lbls)}
itol = {i: l for l, i in ltoi.items()}


def encode_tokens(tl):
    return [stoi.get(t, unk_idx) for t in tl]


def enc_seq(s):
    return encode_tokens(tok(s))


# ---------- augmentation ----------
def augment(tl):
    tl = [t for t in tl]  # copy
    # random span mask
    if len(tl) > 2 and random.random() < 0.4:
        span_len = max(1, int(len(tl) * 0.3))
        start = random.randint(0, len(tl) - span_len)
        tl = tl[:start] + ["<UNK>"] * span_len + tl[start + span_len :]
    # local shuffle
    if len(tl) > 3 and random.random() < 0.3:
        i, j = random.sample(range(len(tl)), 2)
        tl[i], tl[j] = tl[j], tl[i]
    return tl if tl else ["<UNK>"]


# ---------- datasets --------------
class ContrastiveDset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def collate_contrastive(batch):
    aug1 = [encode_tokens(augment(tok(s))) for s in batch]
    aug2 = [encode_tokens(augment(tok(s))) for s in batch]
    comb = aug1 + aug2
    lengths = [len(s) for s in comb]
    mx = max(lengths)
    x = torch.full((len(comb), mx), pad_idx, dtype=torch.long)
    for i, seq in enumerate(comb):
        x[i, : len(seq)] = torch.tensor(seq)
    return x.to(device)


class SupDset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "inp": torch.tensor(enc_seq(self.seqs[idx]), dtype=torch.long),
            "lab": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_sup(batch):
    lens = [len(b["inp"]) for b in batch]
    mx = max(lens)
    x = torch.full((len(batch), mx), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["inp"])] = b["inp"]
    y = torch.stack([b["lab"] for b in batch])
    return {"inp": x.to(device), "lab": y.to(device)}


# ---------- model -----------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, depth=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(512, d_model)  # sequences are short
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, depth)
        self.out_dim = d_model

    def forward(self, x):  # x: (B,L)
        pos = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        h = self.emb(x) + self.pos_emb(pos)
        mask = x == pad_idx
        z = self.tr(h, src_key_padding_mask=mask)
        # mean pool excluding PAD
        mask_inv = (~mask).unsqueeze(-1)
        pooled = (z * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)
        return pooled


class Classifier(nn.Module):
    def __init__(self, enc, num_labels):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(enc.out_dim, num_labels)

    def forward(self, x):
        return self.fc(self.enc(x))


# ---------- contrastive loss ------
def nt_xent(z, temp=0.5):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temp
    N = z.size(0)
    mask = torch.eye(N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    pos = torch.arange(N, device=z.device)
    pos = torch.where(pos < N // 2, pos + N // 2, pos - N // 2)
    return F.cross_entropy(sim, pos)


# ---------- experiment store ------
experiment_data = {
    "SPR": {
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- set random seeds -------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------- build model -----------
enc = TransformerEncoder(len(vocab)).to(device)

# ---------- pre-train -------------
pre_opt = torch.optim.AdamW(enc.parameters(), lr=3e-4)
subsample = int(0.3 * len(spr["train"]))  # use 30 % for speed
pre_loader = DataLoader(
    ContrastiveDset(random.sample(spr["train"]["sequence"], subsample)),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_contrastive,
)
pre_epochs = 2
for ep in range(1, pre_epochs + 1):
    enc.train()
    tot = 0.0
    for xb in pre_loader:
        pre_opt.zero_grad()
        z = enc(xb)
        loss = nt_xent(z)
        loss.backward()
        pre_opt.step()
        tot += loss.item() * xb.size(0)
    l = tot / len(pre_loader.dataset)
    experiment_data["SPR"]["losses"]["pretrain"].append(l)
    print(f"Pretrain epoch {ep}: loss={l:.4f}")

# ---------- fine-tune -------------
clf = Classifier(enc, len(lbls)).to(device)
opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
tr_loader = DataLoader(
    SupDset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_sup
)
val_loader = DataLoader(
    SupDset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_sup
)

best_scwa = -1
best_preds = None
best_true = None
fine_epochs = 4
for ep in range(1, fine_epochs + 1):
    # train
    clf.train()
    tloss = 0.0
    for b in tr_loader:
        opt.zero_grad()
        logits = clf(b["inp"])
        loss = crit(logits, b["lab"])
        loss.backward()
        opt.step()
        tloss += loss.item() * b["lab"].size(0)
    tloss /= len(tr_loader.dataset)
    experiment_data["SPR"]["losses"]["train"].append(tloss)
    # val
    clf.eval()
    vloss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for b in val_loader:
            logits = clf(b["inp"])
            loss = crit(logits, b["lab"])
            vloss += loss.item() * b["lab"].size(0)
            preds += logits.argmax(1).cpu().tolist()
            trues += b["lab"].cpu().tolist()
    vloss /= len(val_loader.dataset)
    experiment_data["SPR"]["losses"]["val"].append(vloss)
    s = swa(spr["dev"]["sequence"], trues, preds)
    c = cwa(spr["dev"]["sequence"], trues, preds)
    sc = scwa_metric(spr["dev"]["sequence"], trues, preds)
    experiment_data["SPR"]["metrics"]["SWA"].append(s)
    experiment_data["SPR"]["metrics"]["CWA"].append(c)
    experiment_data["SPR"]["metrics"]["SCWA"].append(sc)
    experiment_data["SPR"]["epochs"].append(ep)
    print(
        f"Epoch {ep}: validation_loss = {vloss:.4f} | SWA={s:.4f} CWA={c:.4f} SCWA={sc:.4f}"
    )
    if sc > best_scwa:
        best_scwa = sc
        best_preds = preds
        best_true = trues

experiment_data["SPR"]["predictions"] = best_preds
experiment_data["SPR"]["ground_truth"] = best_true

# ---------- save ------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
