import os, random, pathlib, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ---------------- housekeeping -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# --------------- SPR loader (same util as before) ---------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # tiny synthetic fallback

    def _syn(n):
        rows = []
        for _ in range(n):
            L = random.randint(5, 15)
            seq = " ".join(
                random.choice("ABCDE") + random.choice("01234") for _ in range(L)
            )
            rows.append(
                {
                    "id": str(random.randint(0, 1e9)),
                    "sequence": seq,
                    "label": random.randint(0, 1),
                }
            )
        return HFDataset.from_list(rows)

    spr = DatasetDict(train=_syn(4000), dev=_syn(1000), test=_syn(1000))
print({k: len(v) for k, v in spr.items()})

# --------------- basic helpers ------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in spr:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]
V = len(vocab)
MAX_LEN = 40


def encode(seq: str):
    ids = [vocab.get(t, 1) for t in seq.strip().split()[:MAX_LEN]]
    ids += [pad_idx] * (MAX_LEN - len(ids))
    return ids


# ---------- metrics ----------
def count_shape_variety(s):
    return len(set(t[0] for t in s.split() if t))


def count_color_variety(s):
    return len(set(t[1] for t in s.split() if len(t) > 1))


def swa(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, p) if yt == yp) / sum(w)


def cwa(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, p) if yt == yp) / sum(w)


def compwa(seqs, y, p):
    return swa(seqs, y, p) + cwa(seqs, y, p)


# ---------- datasets ----------
class SeqUnlabeled(TorchDataset):
    def __init__(self, hf):
        self.hf = hf

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        return self.hf[i]["sequence"]


class SeqLabeled(TorchDataset):
    def __init__(self, hf):
        self.hf = hf

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        row = self.hf[i]
        return row["sequence"], torch.tensor(row["label"], dtype=torch.long)


def collate_unlab(batch):
    return batch


def collate_lab(batch):
    seqs, labels = zip(*batch)
    ids = torch.tensor([encode(s) for s in seqs], dtype=torch.long)
    return {"sequence": seqs, "input_ids": ids, "labels": torch.stack(labels)}


# --------- data augmentation ----------
def augment(seq: str):
    toks = seq.split()
    if not toks:
        return seq
    op = random.choice(["mask", "shuffle", "drop", "none"])
    if op == "mask":
        idx = random.randrange(len(toks))
        toks[idx] = UNK
    elif op == "shuffle" and len(toks) > 1:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    elif op == "drop" and len(toks) > 2:
        start = random.randrange(len(toks) - 1)
        length = random.randint(1, 2)
        del toks[start : start + length]
    return " ".join(toks)


# ------------- Model -------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=128, nhead=4, nlayer=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.pos = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, ids):
        mask = ids == pad_idx
        x = self.embed(ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        # mean pool
        mask_inv = (~mask).unsqueeze(-1).float()
        h = (x * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1e-6)
        return h


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, encoder, emb_dim, num_cls):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(emb_dim, num_cls)

    def forward(self, ids):
        return self.fc(self.encoder(ids))


# -------- NT-Xent --------------
def nt_xent(z, temperature=0.5):
    z = nn.functional.normalize(z, dim=1)
    N = z.size(0) // 2
    sim = z @ z.T / temperature
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    pos = torch.cat(
        [torch.arange(N, device=z.device) + N, torch.arange(N, device=z.device)]
    )
    pos_sim = sim[torch.arange(2 * N, device=z.device), pos]
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()


# ---------- training settings ----------
EMB_DIM = 128
PRE_EPOCHS = 2
FT_EPOCHS = 4
BATCH_CTR = 256
BATCH_FT = 128
NUM_CLASSES = len(set(spr["train"]["label"]))

# ---------- loaders ----------
unlab_loader = DataLoader(
    SeqUnlabeled(spr["train"]),
    batch_size=BATCH_CTR,
    shuffle=True,
    collate_fn=collate_unlab,
)

train_loader = DataLoader(
    SeqLabeled(spr["train"]), batch_size=BATCH_FT, shuffle=True, collate_fn=collate_lab
)
dev_loader = DataLoader(
    SeqLabeled(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_lab
)

# ---------- build model ----------
encoder = TransformerEncoder(V, EMB_DIM).to(device)
proj = ProjectionHead(EMB_DIM).to(device)
opt_ct = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=3e-3)

print("\nContrastive pre-training")
for ep in range(1, PRE_EPOCHS + 1):
    encoder.train()
    proj.train()
    tot = 0
    for seqs in unlab_loader:
        v1 = [augment(s) for s in seqs]
        v2 = [augment(s) for s in seqs]
        ids = torch.tensor([encode(s) for s in v1 + v2], dtype=torch.long).to(device)
        opt_ct.zero_grad()
        h = encoder(ids)
        z = proj(h)
        loss = nt_xent(z)
        loss.backward()
        opt_ct.step()
        tot += loss.item()
    print(f"pre-epoch {ep}: contrastive_loss={tot/len(unlab_loader):.4f}")

# -------- fine-tune ------------
classifier = Classifier(encoder, EMB_DIM, NUM_CLASSES).to(device)
opt_ft = torch.optim.Adam(classifier.parameters(), lr=3e-3)
crit = nn.CrossEntropyLoss()

experiment_data = {
    "transformer_simclr": {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "CompWA": []},
        "predictions": [],
        "ground_truth": [],
    }
}

print("\nSupervised fine-tuning")
for ep in range(1, FT_EPOCHS + 1):
    classifier.train()
    tr_loss = 0
    for batch in train_loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt_ft.zero_grad()
        logits = classifier(batch_t["input_ids"])
        loss = crit(logits, batch_t["labels"])
        loss.backward()
        opt_ft.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)
    experiment_data["transformer_simclr"]["losses"]["train"].append((ep, tr_loss))

    # ---- validation -----
    classifier.eval()
    val_loss, seqs, gt, pr = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = classifier(batch_t["input_ids"])
            val_loss += crit(logits, batch_t["labels"]).item()
            preds = logits.argmax(-1).cpu().tolist()
            pr.extend(preds)
            gt.extend(batch["labels"].tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(dev_loader)
    SWA = swa(seqs, gt, pr)
    CWA = cwa(seqs, gt, pr)
    Comp = SWA + CWA
    experiment_data["transformer_simclr"]["losses"]["val"].append((ep, val_loss))
    experiment_data["transformer_simclr"]["metrics"]["SWA"].append((ep, SWA))
    experiment_data["transformer_simclr"]["metrics"]["CWA"].append((ep, CWA))
    experiment_data["transformer_simclr"]["metrics"]["CompWA"].append((ep, Comp))
    experiment_data["transformer_simclr"]["predictions"].append((ep, pr))
    experiment_data["transformer_simclr"]["ground_truth"].append((ep, gt))

    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={SWA:.3f} "
        f"CWA={CWA:.3f} CompWA={Comp:.3f}"
    )

# ---- save ----
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
