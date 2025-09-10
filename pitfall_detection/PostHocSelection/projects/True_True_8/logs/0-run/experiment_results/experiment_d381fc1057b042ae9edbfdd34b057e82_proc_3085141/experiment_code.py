import os, pathlib, random, math, time, string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# -------------------- house-keeping --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- load SPR_BENCH -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):  # treat each csv as a stand-alone split
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # ---------- tiny synthetic fallback -------------

    def _synth(n, max_len=10):
        rows = []
        for i in range(n):
            L = random.randint(4, max_len)
            seq, label = [], 0
            for _ in range(L):
                sh, co = random.choice("ABCDE"), random.choice("01234")
                seq.append(sh + co)
                label ^= (ord(sh) + int(co)) & 1
            rows.append({"id": str(i), "sequence": " ".join(seq), "label": label})
        return HFDataset.from_list(rows)

    spr = DatasetDict(train=_synth(4000), dev=_synth(1000), test=_synth(1000))
print({k: len(v) for k, v in spr.items()})

# -------------------- vocabulary -----------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in spr.values():
    for s in split["sequence"]:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]
VOCAB_SIZE = len(vocab)
MAX_LEN = 40


def encode(seq: str):
    ids = [vocab.get(t, 1) for t in seq.split()][:MAX_LEN]
    ids += [pad_idx] * (MAX_LEN - len(ids))
    return ids


# -------------------- metrics --------------------------
def _shape_variety(s):
    return len({t[0] for t in s.split()})


def _color_variety(s):
    return len({t[1] for t in s.split() if len(t) > 1})


def SWA(seqs, y_t, y_p):
    w = [_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w)


def CWA(seqs, y_t, y_p):
    w = [_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w)


def CompWA(seqs, y_t, y_p):
    w = [_shape_variety(s) + _color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w)


# -------------------- augmentations --------------------
def aug_mask(tokens, p=0.3):
    return [tok if random.random() > p else UNK for tok in tokens]


def aug_shuffle(tokens, frac=0.3):
    n_swap = max(1, int(len(tokens) * frac))
    tok = tokens[:]
    for _ in range(n_swap):
        i, j = random.sample(range(len(tok)), 2)
        tok[i], tok[j] = tok[j], tok[i]
    return tok


def two_views(seq: str):
    toks = seq.split()
    v1 = aug_mask(aug_shuffle(toks))
    v2 = aug_mask(aug_shuffle(toks))
    return " ".join(v1), " ".join(v2)


# -------------------- torch dataset --------------------
class ContrastiveSPRTorch(TorchDataset):
    def __init__(self, hfds: HFDataset, take=None):
        self.ds = hfds if take is None else hfds.shuffle(seed=0).select(range(take))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        v1, v2 = two_views(row["sequence"])
        return {
            "seq": row["sequence"],
            "v1": torch.tensor(encode(v1), dtype=torch.long),
            "v2": torch.tensor(encode(v2), dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


def collate(batch):
    return {
        k: ([b[k] for b in batch] if k == "seq" else torch.stack([b[k] for b in batch]))
        for k in batch[0]
    }


# -------------------- model ----------------------------
class Encoder(nn.Module):
    def __init__(self, embed_dim=128, hid=128):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.project = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, hid)
        )
        self.feat_dim = hid * 2  # <-- expose pooled feature size

    def forward(self, x):
        mask = x != pad_idx
        emb = self.emb(x)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (num_layers*dirs,B,hid)
        feat = torch.cat([h[-2], h[-1]], dim=1)  # B, hid*2
        z = self.project(feat)
        return nn.functional.normalize(z, dim=-1), feat


class SPRModel(nn.Module):
    def __init__(self, enc: Encoder):
        super().__init__()
        self.enc = enc
        self.cls = nn.Linear(enc.feat_dim, 2)  # correct input dim

    def forward(self, x):
        z, feat = self.enc(x)
        return z, self.cls(feat)


# ---------------- contrastive loss ---------------------
def info_nce(z1, z2, temp=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2B,d
    sim = torch.matmul(z, z.t()) / temp
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], 0)
    return nn.functional.cross_entropy(sim, targets)


# ---------------- training setup -----------------------
BATCH = 256
train_loader = DataLoader(
    ContrastiveSPRTorch(spr["train"], take=8000),
    batch_size=BATCH,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    ContrastiveSPRTorch(spr["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate
)

model = SPRModel(Encoder()).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
ce_loss_fn = nn.CrossEntropyLoss()
LAMBDA_CON = 0.7
EPOCHS = 8

# --------------- experiment record ---------------------
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- training loop ------------------------
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    t_loss, steps = 0.0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt.zero_grad()
        z1, _ = model.enc(batch["v1"])
        z2, _ = model.enc(batch["v2"])
        con_loss = info_nce(z1, z2)
        _, logits = model(batch["v1"])
        ce_loss = ce_loss_fn(logits, batch["label"])
        loss = LAMBDA_CON * con_loss + (1 - LAMBDA_CON) * ce_loss
        loss.backward()
        opt.step()
        t_loss += loss.item()
        steps += 1
    train_loss = t_loss / steps
    experiment_data["SPR"]["losses"]["train"].append((epoch, train_loss))

    # ---- validation ----
    model.eval()
    v_loss = 0.0
    seqs = []
    gts = []
    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            seqs.extend(batch["seq"])
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            _, logits = model(batch["v1"])
            loss = ce_loss_fn(logits, batch["label"])
            v_loss += loss.item()
            gts.extend(batch["label"].cpu().tolist())
            preds.extend(logits.argmax(-1).cpu().tolist())
    v_loss /= len(dev_loader)
    swa, cwa, comp = (
        SWA(seqs, gts, preds),
        CWA(seqs, gts, preds),
        CompWA(seqs, gts, preds),
    )
    cowa = (swa + cwa) / 2
    experiment_data["SPR"]["losses"]["val"].append((epoch, v_loss))
    experiment_data["SPR"]["metrics"]["val"].append((epoch, swa, cwa, cowa, comp))
    experiment_data["SPR"]["predictions"].append((epoch, preds))
    experiment_data["SPR"]["ground_truth"].append((epoch, gts))

    print(
        f"Epoch {epoch}: validation_loss = {v_loss:.4f} | "
        f"SWA {swa:.4f} CWA {cwa:.4f} CoWA {cowa:.4f} CompWA {comp:.4f}"
    )

# ---------------- save everything ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
