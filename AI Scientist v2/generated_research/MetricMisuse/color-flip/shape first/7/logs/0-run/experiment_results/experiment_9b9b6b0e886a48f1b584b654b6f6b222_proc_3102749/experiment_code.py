import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------ working dir & device ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ reproducibility ----------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ------------------ locate SPR_BENCH ---------------------
def find_spr() -> pathlib.Path:
    for p in [
        "SPR_BENCH",
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        p = pathlib.Path(p)
        if p.joinpath("train.csv").exists():
            return p
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_DIR = find_spr()
print(f"Found SPR_BENCH at {DATA_DIR}")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr(DATA_DIR)


# ------------------ metrics ------------------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def SWA(seqs, y, yh):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yi == yhi else 0 for wi, yi, yhi in zip(w, y, yh)) / sum(w)


def CWA(seqs, y, yh):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yi == yhi else 0 for wi, yi, yhi in zip(w, y, yh)) / sum(w)


# ------------------ vocab --------------------------------
all_toks = set(tok for s in spr["train"]["sequence"] for tok in s.split())
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_toks))}
PAD = 0
VOCAB = len(tok2id) + 1


def encode(s):
    return [tok2id[t] for t in s.split()]


NUM_CLASSES = len(set(spr["train"]["label"]))


# ------------------ data augmentation --------------------
def augment(ids, drop_p=0.2, shuffle_span=3):
    ids = [i for i in ids if i != PAD]
    # token dropout
    kept = [i for i in ids if random.random() > drop_p]
    if not kept:
        kept = [random.choice(ids)]
    # local shuffle
    for i in range(0, len(kept), shuffle_span):
        seg = kept[i : i + shuffle_span]
        random.shuffle(seg)
        kept[i : i + shuffle_span] = seg
    return kept


# ------------------ torch datasets -----------------------
class SPRContrastSet(Dataset):
    def __init__(self, split):
        self.raw = split["sequence"]
        self.lbl = split["label"]
        self.encoded = [encode(s) for s in self.raw]

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        ids = self.encoded[idx]
        view1 = augment(ids)
        view2 = augment(ids)
        return {
            "v1": torch.tensor(view1),
            "v2": torch.tensor(view2),
            "orig": torch.tensor(ids),
            "label": torch.tensor(self.lbl[idx]),
            "raw_seq": self.raw[idx],
        }


def pad_collate(batch, key):
    mx = max(len(b[key]) for b in batch)
    return torch.stack([F.pad(b[key], (0, mx - len(b[key])), value=PAD) for b in batch])


def collate_contrast(batch):
    return {
        "v1": pad_collate(batch, "v1"),
        "v2": pad_collate(batch, "v2"),
        "orig": pad_collate(batch, "orig"),
        "label": torch.tensor([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_set = SPRContrastSet(spr["train"])
dev_set = SPRContrastSet(spr["dev"])
train_loader = DataLoader(
    train_set, batch_size=256, shuffle=True, collate_fn=collate_contrast
)
dev_loader = DataLoader(
    dev_set, batch_size=512, shuffle=False, collate_fn=collate_contrast
)


# ------------------ model --------------------------------
class Encoder(nn.Module):
    def __init__(self, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb, padding_idx=PAD)
        self.rnn = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)

    def forward(self, x):
        e = self.emb(x)
        lens = (x != PAD).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.rnn(packed)
        return torch.cat([h[-2], h[-1]], 1)  # N, 2*hid


class SimCLR(nn.Module):
    def __init__(self, enc, proj_dim=128):
        super().__init__()
        self.enc = enc
        self.proj = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        z = self.enc(x)
        return F.normalize(self.proj(z), dim=1)


def nt_xent(z1, z2, t=0.07):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), -1) / t
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.arange(B, device=z.device)
    targets = torch.cat([pos + B, pos])
    return F.cross_entropy(sim, targets)


class Classifier(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        return self.fc(self.enc(x))


# ------------------ experiment container -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------ contrastive pre-train -----------------
enc = Encoder().to(device)
model_con = SimCLR(enc).to(device)
opt_con = torch.optim.Adam(model_con.parameters(), lr=1e-3)

epochs_con = 3
for ep in range(1, epochs_con + 1):
    model_con.train()
    tot = 0
    n = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt_con.zero_grad()
        z1 = model_con(batch["v1"])
        z2 = model_con(batch["v2"])
        loss = nt_xent(z1, z2)
        loss.backward()
        opt_con.step()
        tot += loss.item()
        n += 1
    print(f"Contrastive Epoch {ep}: loss={tot/n:.4f}")

# ------------------ supervised fine-tune ------------------
model_sup = Classifier(enc).to(device)  # re-use encoder weights
opt_sup = torch.optim.Adam(model_sup.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def evaluate(loader):
    model_sup.eval()
    preds = []
    gts = []
    seqs = []
    with torch.no_grad():
        for b in loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in b.items()
            }
            out = model_sup(b["orig"])
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(b["label"].cpu().tolist())
            seqs.extend(b["raw_seq"])
    swa = SWA(seqs, gts, preds)
    cwa = CWA(seqs, gts, preds)
    dawa = 0.5 * (swa + cwa)
    return preds, gts, swa, cwa, dawa


epochs_sup = 5
for ep in range(1, epochs_sup + 1):
    model_sup.train()
    tot = 0
    n = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt_sup.zero_grad()
        logits = model_sup(batch["orig"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        opt_sup.step()
        tot += loss.item()
        n += 1
    tr_loss = tot / n
    preds, gts, swa, cwa, dawa = evaluate(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, swa, cwa, dawa))
    if ep == epochs_sup:
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = gts
    print(
        f"Epoch {ep}: validation_loss = NA | SWA={swa:.4f} CWA={cwa:.4f} DAWA={dawa:.4f}"
    )

# ------------------ save ---------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
