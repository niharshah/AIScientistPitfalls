import os, random, string, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# --------------------------------------------------- experiment data dict
experiment_data = {
    "freeze_encoder": {
        "SPR": {
            "contrastive_pretrain": {"losses": []},
            "fine_tune": {
                "losses": {"train": [], "val": []},
                "metrics": {"SWA": [], "CWA": [], "CompWA": []},
                "predictions": [],
                "ground_truth": [],
            },
        }
    }
}

# ---------------------------------------------------------- paths & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------------ load / build data
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


def build_synthetic_dataset(n_tr=2000, n_dev=500, n_test=500, max_len=10):
    def _row():
        L, seq, label = random.randint(4, max_len), [], 0
        for _ in range(L):
            sh, co = random.choice("ABCDE"), random.choice("01234")
            seq.append(sh + co)
            label ^= (ord(sh) + int(co)) & 1
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": label,
        }

    def _many(n):
        return [_row() for _ in range(n)]

    return DatasetDict(
        train=HFDataset.from_list(_many(n_tr)),
        dev=HFDataset.from_list(_many(n_dev)),
        test=HFDataset.from_list(_many(n_test)),
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_synthetic_dataset()
print({k: len(v) for k, v in spr.items()})

# --------------------------------------------------------- vocab / encode
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx, MAX_LEN = vocab[PAD], 40


def encode(seq, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.split()][:max_len]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# --------------------------------------------------------- metrics helpers
def count_shape_variety(sequence):
    return len({t[0] for t in sequence.split()})


def count_color_variety(sequence):
    return len({t[1] for t in sequence.split() if len(t) > 1})


def _weighted_acc(seqs, y_t, y_p, func):
    w = [func(s) for s in seqs]
    tot = sum(w)
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / tot if tot else 0.0


def shape_weighted_accuracy(s, y_t, y_p):
    return _weighted_acc(s, y_t, y_p, count_shape_variety)


def color_weighted_accuracy(s, y_t, y_p):
    return _weighted_acc(s, y_t, y_p, count_color_variety)


def complexity_weighted_accuracy(s, y_t, y_p):
    return _weighted_acc(
        s, y_t, y_p, lambda seq: count_shape_variety(seq) + count_color_variety(seq)
    )


# ------------------------------------------------------- augmentations
def shape_rename(seq):
    toks = seq.split()
    mapping = {s: random.choice(string.ascii_uppercase) for s in {t[0] for t in toks}}
    return " ".join([mapping[t[0]] + t[1:] for t in toks])


def color_rename(seq):
    toks = seq.split()
    mapping = {
        c: random.choice("0123456789") for c in {t[1] for t in toks if len(t) > 1}
    }
    return " ".join([t[0] + mapping.get(t[1], t[1]) for t in toks])


def token_dropout(seq, p=0.15):
    toks = [t for t in seq.split() if random.random() > p]
    return " ".join(toks if toks else seq.split())


def augment(seq):
    if random.random() < 0.4:
        seq = shape_rename(seq)
    if random.random() < 0.4:
        seq = color_rename(seq)
    if random.random() < 0.3:
        seq = token_dropout(seq)
    return seq


# ----------------------------------------------------------- torch datasets
class ContrastiveSPRDataset(TorchDataset):
    def __init__(self, hf):
        self.ds = hf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[idx]["sequence"]
        v1, v2 = augment(s), augment(s)
        return torch.tensor(encode(v1)), torch.tensor(encode(v2))


class ClassificationSPRDataset(TorchDataset):
    def __init__(self, hf):
        self.ds = hf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        r = self.ds[idx]
        return (
            torch.tensor(encode(r["sequence"])),
            torch.tensor(r["label"]),
            r["sequence"],
        )


def collate_contrastive(batch):
    v1 = torch.stack([b[0] for b in batch])
    v2 = torch.stack([b[1] for b in batch])
    return {"view1": v1, "view2": v2}


def collate_classification(batch):
    ids = torch.stack([b[0] for b in batch])
    lbl = torch.stack([b[1] for b in batch])
    seqs = [b[2] for b in batch]
    return {"input_ids": ids, "labels": lbl, "sequence": seqs}


# ------------------------------------------------------------- model
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)

    def forward(self, x):
        emb = self.emb(x)
        mask = (x != pad_idx).float().unsqueeze(-1)
        packed, _ = self.gru(emb)
        return (packed * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, T=0.07):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.t()) / T
    mask = (~torch.eye(2 * N, dtype=torch.bool, device=z.device)).float()
    sim -= 1e9 * (1 - mask)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])
    return nn.CrossEntropyLoss()(sim, labels)


# ---------------------------------------------------- contrastive pretrain
BATCH_C, PRE_EPOCHS = 256, 2
train_subset = spr["train"].shuffle(seed=0).select(range(min(5000, len(spr["train"]))))
c_loader = DataLoader(
    ContrastiveSPRDataset(train_subset),
    batch_size=BATCH_C,
    shuffle=True,
    collate_fn=collate_contrastive,
)
encoder, projector = Encoder(len(vocab)).to(device), Projector(512).to(device)
opt_pre = torch.optim.Adam(
    list(encoder.parameters()) + list(projector.parameters()), lr=3e-3
)

print("\n--- Contrastive pre-training ---")
for ep in range(1, PRE_EPOCHS + 1):
    encoder.train()
    projector.train()
    run = 0.0
    for batch in c_loader:
        v1, v2 = batch["view1"].to(device), batch["view2"].to(device)
        loss = nt_xent_loss(projector(encoder(v1)), projector(encoder(v2)))
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        run += loss.item()
    avg = run / len(c_loader)
    experiment_data["freeze_encoder"]["SPR"]["contrastive_pretrain"]["losses"].append(
        (ep, avg)
    )
    print(f"Pre-epoch {ep}: loss={avg:.4f}")


# -------------------------------------------------------- classifier (frozen encoder)
class Classifier(nn.Module):
    def __init__(self, enc, num_cls=2):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(512, num_cls)

    def forward(self, x):
        return self.fc(self.enc(x))


# freeze encoder
for p in encoder.parameters():
    p.requires_grad = False
model = Classifier(encoder).to(device)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)  # only head
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    ClassificationSPRDataset(spr["train"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_classification,
)
dev_loader = DataLoader(
    ClassificationSPRDataset(spr["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classification,
)

FINE_EPOCHS = 5
print("\n--- Fine-tuning with frozen encoder ---")
for ep in range(1, FINE_EPOCHS + 1):
    # train
    model.train()
    run_loss = 0.0
    for batch in train_loader:
        ids, lbl = batch["input_ids"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(ids)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    tr_loss = run_loss / len(train_loader)
    experiment_data["freeze_encoder"]["SPR"]["fine_tune"]["losses"]["train"].append(
        (ep, tr_loss)
    )
    # validation
    model.eval()
    val_loss = 0.0
    preds, gts, seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids, lbl = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(ids)
            val_loss += criterion(logits, lbl).item()
            p = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            gts.extend(batch["labels"].tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(dev_loader)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    comp = complexity_weighted_accuracy(seqs, gts, preds)
    fd = experiment_data["freeze_encoder"]["SPR"]["fine_tune"]
    fd["losses"]["val"].append((ep, val_loss))
    fd["metrics"]["SWA"].append((ep, swa))
    fd["metrics"]["CWA"].append((ep, cwa))
    fd["metrics"]["CompWA"].append((ep, comp))
    fd["predictions"].append((ep, preds))
    fd["ground_truth"].append((ep, gts))
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f}  SWA={swa:.4f} CWA={cwa:.4f} CompWA={comp:.4f}"
    )

# ----------------------------------------------------- save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
