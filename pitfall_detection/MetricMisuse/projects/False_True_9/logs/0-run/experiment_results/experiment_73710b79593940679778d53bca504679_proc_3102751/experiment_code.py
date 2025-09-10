import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- folder, device ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- deterministic -----------------------------------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ---------- locate SPR_BENCH ---------------------------------------------------
def locate_spr():
    cand = [
        os.environ.get("SPR_DATA_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).joinpath("train.csv").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("SPR_BENCH not found")


DATA_PATH = locate_spr()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------- metrics helpers ---------------------------------------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def SWA(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w)


def CWA(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w)


def DAWA(swa, cwa):
    return (swa + cwa) / 2


# ---------- load dataset -------------------------------------------------------
def load_spr(root):
    def _l(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


spr = load_spr(DATA_PATH)

# ---------- vocabulary ---------------------------------------------------------
tok_set = set(tok for ex in spr["train"] for tok in ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(tok_set))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))


# ---------- dataset ------------------------------------------------------------
class SPRSet(Dataset):
    def __init__(self, split, augment=False):
        self.seq = split["sequence"]
        self.lab = split["label"]
        self.aug = augment
        self.enc = [encode(s) for s in self.seq]

    def __len__(self):
        return len(self.lab)

    def _mask(self, tok_ids):
        ids = tok_ids.copy()
        for i in range(len(ids)):
            if ids[i] != PAD_ID and random.random() < 0.15:
                ids[i] = PAD_ID  # simple mask with PAD
        return ids

    def __getitem__(self, idx):
        ids = self.enc[idx]
        if self.aug:
            view1 = self._mask(ids)
            view2 = self._mask(ids)
            return {
                "view1": torch.tensor(view1),
                "view2": torch.tensor(view2),
                "label": torch.tensor(self.lab[idx]),
                "raw": self.seq[idx],
            }
        return {
            "ids": torch.tensor(ids),
            "label": torch.tensor(self.lab[idx]),
            "raw": self.seq[idx],
        }


def pad_stack(seqs, maxlen):
    out = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate_train(batch):
    v1 = [b["view1"] for b in batch]
    v2 = [b["view2"] for b in batch]
    mlen = max(max(len(x) for x in v1), max(len(x) for x in v2))
    return {
        "view1": pad_stack(v1, mlen),
        "view2": pad_stack(v2, mlen),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


def collate_eval(batch):
    ids = [b["ids"] for b in batch]
    m = max(len(x) for x in ids)
    return {
        "ids": pad_stack(ids, m),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


train_loader = DataLoader(
    SPRSet(spr["train"], augment=True),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_train,
)
dev_loader = DataLoader(
    SPRSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_eval
)


# ---------- model ----------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SPRModel(nn.Module):
    def __init__(self, vocab, emb_dim=64, hidden=128, nlayers=2, nhead=8, classes=10):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(emb_dim, classes)

    def represent(self, x):
        mask = x == PAD_ID
        h = self.emb(x)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_invert = (~mask).unsqueeze(-1)
        summed = (h * mask_invert).sum(1)
        lens = mask_invert.sum(1).clamp(min=1)
        return summed / lens  # mean pooling

    def forward(self, x):
        rep = self.represent(x)
        return self.fc(rep), rep


# ---------- supervised contrastive loss -----------------------------------------
def supcon_loss(features, labels, temp=0.07):
    # features: [2B, D], labels: [2B]
    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)
    # logits adjustment
    logits = (
        anchor_dot_contrast - torch.max(anchor_dot_contrast, dim=1, keepdim=True).values
    )
    exp_logits = torch.exp(logits) * (1 - torch.eye(labels.size(0)).to(device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()
    return loss


# ---------- experiment container -------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- training -------------------------------------------------------------
model = SPRModel(vocab_size, classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 6
for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    tr_loss_cum = 0
    nb = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits1, rep1 = model(batch["view1"])
        logits2, rep2 = model(batch["view2"])
        ce = criterion(logits1, batch["label"])
        reps = torch.cat([rep1, rep2], dim=0)
        lbls = torch.cat([batch["label"], batch["label"]], dim=0)
        scl = supcon_loss(reps, lbls)
        loss = ce + 0.1 * scl
        loss.backward()
        optimizer.step()
        tr_loss_cum += loss.item()
        nb += 1
    train_loss = tr_loss_cum / nb
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))
    # ---- validate ----
    model.eval()
    vl = 0
    nb = 0
    preds = []
    labs = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits, _ = model(batch["ids"])
            loss = criterion(logits, batch["label"])
            vl += loss.item()
            nb += 1
            p = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            labs.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw"])
    val_loss = vl / nb
    swa = SWA(seqs, labs, preds)
    cwa = CWA(seqs, labs, preds)
    dawa = DAWA(swa, cwa)
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, swa, cwa, dawa))
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} DAWA={dawa:.4f}"
    )
    if epoch == epochs:
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = labs

# ---------- save -----------------------------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to working/experiment_data.npy")
