import os, pathlib, random, math, time, string
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset as HFDataset, DatasetDict, load_dataset

# ----------------- housekeeping -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# --------------- data loading -------------------
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


def build_synthetic_dataset(n=2000) -> DatasetDict:
    def _gen_row():
        l = random.randint(4, 12)
        seq, label = [], 0
        for _ in range(l):
            sh, co = random.choice("ABCDE"), random.choice("01234")
            seq.append(sh + co)
            label ^= (ord(sh) + int(co)) & 1
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": label,
        }

    rows = [_gen_row() for _ in range(n)]
    tr, dv = int(0.8 * n), int(0.1 * n)
    return DatasetDict(
        train=HFDataset.from_list(rows[:tr]),
        dev=HFDataset.from_list(rows[tr : tr + dv]),
        test=HFDataset.from_list(rows[tr + dv :]),
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_synthetic_dataset()
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# --------------- vocabulary ----------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in spr.values():
    for seq in split["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
MAX_LEN = 40


def encode(seq: str):
    ids = [vocab.get(t, 1) for t in seq.strip().split()[:MAX_LEN]]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


# --------------- metrics ------------------------
def count_shape_variety(s):
    return len({t[0] for t in s.split()})


def count_color_variety(s):
    return len({t[1] for t in s.split() if len(t) > 1})


def SWA(seq, y, p):
    w = [count_shape_variety(s) for s in seq]
    c = [wi if yt == pt else 0 for wi, yt, pt in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def CWA(seq, y, p):
    w = [count_color_variety(s) for s in seq]
    c = [wi if yt == pt else 0 for wi, yt, pt in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def CompWA(seq, y, p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seq]
    c = [wi if yt == pt else 0 for wi, yt, pt in zip(w, y, p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# --------------- augmentations ------------------
def mask_tokens(tokens, p=0.3):
    return [tok if random.random() > p else UNK for tok in tokens]


def local_shuffle(tokens, k=3):
    if len(tokens) <= k:
        return tokens
    i = random.randint(0, len(tokens) - k)
    sub = tokens[i : i + k]
    random.shuffle(sub)
    return tokens[:i] + sub + tokens[i + k :]


def augment(seq):
    toks = seq.split()
    if random.random() < 0.5:
        toks = mask_tokens(toks)
    else:
        toks = local_shuffle(toks)
    return " ".join(toks)


# --------------- torch datasets -----------------
class ContrastiveDS(TorchDataset):
    def __init__(self, hf, size=4000):
        self.rows = [hf[i] for i in range(min(size, len(hf)))]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]["sequence"]
        return (
            torch.tensor(encode(augment(s)), dtype=torch.long),
            torch.tensor(encode(augment(s)), dtype=torch.long),
        )


def collate_pairs(batch):
    x1 = torch.stack([b[0] for b in batch])
    x2 = torch.stack([b[1] for b in batch])
    return {"x1": x1, "x2": x2}


class ClassifyDS(TorchDataset):
    def __init__(self, hf, size=None):
        self.rows = [hf[i] for i in range(len(hf) if size is None else size)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "sequence": row["sequence"],
            "input_ids": torch.tensor(encode(row["sequence"]), dtype=torch.long),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


def collate_cls(batch):
    return {
        "sequence": [b["sequence"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# --------------- models -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        summed = (self.emb(x) * mask).sum(1)
        lens = mask.sum(1).clamp(min=1e-6)
        return summed / lens


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class Classifier(nn.Module):
    def __init__(self, enc, hidden):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc(self.enc(x))


# --------------- fixed NT-Xent ------------------
def nt_xent(z1, z2, temperature=0.5):
    """
    SimCLR NT-Xent with proper masking (no column deletion)
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2B,D
    sim = torch.matmul(z, z.T) / temperature  # 2B,2B
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)  # mask self-sim
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], 0)  # positive indices
    loss = F.cross_entropy(sim, targets)
    return loss


# --------------- experiment store ---------------
experiment_data = {
    "contrastive_cls": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------- contrastive pre-training -------
ENC_DIM, PROJ_DIM = 128, 64
CONTR_EPOCHS = 3
BATCH = 256
encoder = Encoder(len(vocab), ENC_DIM).to(device)
proj = ProjectionHead(ENC_DIM, PROJ_DIM).to(device)
opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=3e-3)

contrast_loader = DataLoader(
    ContrastiveDS(spr["train"]),
    batch_size=BATCH,
    shuffle=True,
    collate_fn=collate_pairs,
)

print("\n--- Contrastive pre-training ---")
for ep in range(1, CONTR_EPOCHS + 1):
    encoder.train()
    proj.train()
    run = 0.0
    for batch in contrast_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z1, z2 = proj(encoder(batch["x1"])), proj(encoder(batch["x2"]))
        loss = nt_xent(z1, z2, 0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += loss.item()
    print(f"Contrastive epoch {ep}: loss={run/len(contrast_loader):.4f}")

# --------------- classification fine-tune -------
CLS_EPOCHS = 5
BATCH = 128
clf = Classifier(encoder, ENC_DIM).to(device)
clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    ClassifyDS(spr["train"], size=4000),
    batch_size=BATCH,
    shuffle=True,
    collate_fn=collate_cls,
)
dev_loader = DataLoader(
    ClassifyDS(spr["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate_cls
)

print("\n--- Classification fine-tuning ---")
for ep in range(1, CLS_EPOCHS + 1):
    # ---- train ----
    clf.train()
    run_loss = 0.0
    for batch in train_loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = clf(batch_t["input_ids"])
        loss = criterion(logits, batch_t["labels"])
        clf_opt.zero_grad()
        loss.backward()
        clf_opt.step()
        run_loss += loss.item()
    train_loss = run_loss / len(train_loader)
    experiment_data["contrastive_cls"]["losses"]["train"].append((ep, train_loss))

    # ---- validate ----
    clf.eval()
    val_loss = 0.0
    seqs = []
    gts = []
    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = clf(batch_t["input_ids"])
            val_loss += criterion(logits, batch_t["labels"]).item()
            p = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            gts.extend(batch["labels"].tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(dev_loader)
    swa, cwa, comp = (
        SWA(seqs, gts, preds),
        CWA(seqs, gts, preds),
        CompWA(seqs, gts, preds),
    )
    experiment_data["contrastive_cls"]["losses"]["val"].append((ep, val_loss))
    experiment_data["contrastive_cls"]["metrics"]["train"].append((ep, swa, cwa, comp))
    experiment_data["contrastive_cls"]["predictions"].append((ep, preds))
    experiment_data["contrastive_cls"]["ground_truth"].append((ep, gts))

    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | "
        f"SWA={swa:.4f} CWA={cwa:.4f} CompWA={comp:.4f}"
    )

# --------------- save everything ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
