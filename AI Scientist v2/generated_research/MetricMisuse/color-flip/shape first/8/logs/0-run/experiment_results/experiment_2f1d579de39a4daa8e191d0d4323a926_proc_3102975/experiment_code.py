# ---------------------------------------------
# Mask-Only Views Ablation for SPR Contrasting
# ---------------------------------------------
import os, random, math, pathlib, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- mandatory working directory ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device -------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper to load SPR_BENCH -------------------------
def load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    def _load(file_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / file_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


def try_load_spr():
    for cand in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (cand / "train.csv").exists():
            print(f"Found SPR_BENCH at {cand}")
            return load_spr_bench(cand)
    return None


real_dset = try_load_spr()

# ---------- synthetic fall-back --------------------------------
shapes, colors = list("RSTUVWXYZ"), list("ABCDEFGH")


def rnd_token():
    return random.choice(shapes) + random.choice(colors)


def rnd_seq():
    return " ".join(rnd_token() for _ in range(random.randint(4, 12)))


def make_split(n):
    return [
        {"id": i, "sequence": rnd_seq(), "label": random.randint(0, 3)}
        for i in range(n)
    ]


if real_dset is None:
    print("Real data not found – using synthetic fallback.")
    real_dset = {
        "train": make_split(6000),
        "dev": make_split(1200),
        "test": make_split(1200),
    }


# ---------- CCWA metric helpers -------------------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def ccwa_metric(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ---------- vocabulary ----------------------------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    vocab = {tok for r in rows for tok in r["sequence"].split()}
    itos = [PAD, UNK, MASK] + sorted(vocab)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size, MAX_LEN = len(stoi), 20


def encode(seq: str):
    ids = [stoi.get(tok, stoi[UNK]) for tok in seq.split()][:MAX_LEN]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


# ---------- datasets (mask-only augmentation) ------------------
class ContrastiveSPRMaskOnly(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def _augment(self, ids):
        toks = [t for t in ids if t != stoi[PAD]]
        toks = [stoi[MASK] if random.random() < 0.15 else t for t in toks]
        aug_seq = " ".join(itos[t] for t in toks)
        return torch.tensor(encode(aug_seq), dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        base_ids = torch.tensor(encode(self.rows[idx]["sequence"]), dtype=torch.long)
        return {"view1": self._augment(base_ids), "view2": self._augment(base_ids)}


class SupervisedSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "ids": torch.tensor(encode(r["sequence"]), dtype=torch.long),
            "label": torch.tensor(r["label"], dtype=torch.long),
            "seq": r["sequence"],
        }


# ---------- model ---------------------------------------------
class TransEncoder(nn.Module):
    def __init__(self, vocab, d_model=96, nhead=6, nlayers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_model, 128)

    def forward(self, x):
        x = self.emb(x.to(device)) + self.pos[: x.size(1)].unsqueeze(0)
        h = self.transformer(x)
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------- SimCLR loss ---------------------------------------
def simclr_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * B, device=z.device, dtype=torch.bool), -1e9)
    positives = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, positives)


# ---------- experiment bookkeeping ----------------------------
experiment_data = {
    "mask_only": {
        "SPR_BENCH": {
            "metrics": {"train_CCWA": [], "val_CCWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ---------- data loaders --------------------------------------
BATCH_PRE, BATCH_FT = 256, 256
pre_ds = ContrastiveSPRMaskOnly(real_dset["train"])
train_ds = SupervisedSPR(real_dset["train"])
dev_ds = SupervisedSPR(real_dset["dev"])

pre_dl = DataLoader(pre_ds, batch_size=BATCH_PRE, shuffle=True, drop_last=True)
train_dl = DataLoader(train_ds, batch_size=BATCH_FT, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=BATCH_FT, shuffle=False)

# ---------- pre-training --------------------------------------
encoder = TransEncoder(vocab_size).to(device)
opt_pre = torch.optim.Adam(encoder.parameters(), lr=1e-3)
PRE_EPOCHS = 6

for ep in range(1, PRE_EPOCHS + 1):
    encoder.train()
    total = 0.0
    for batch in pre_dl:
        v1, v2 = batch["view1"].to(device), batch["view2"].to(device)
        loss = simclr_loss(encoder(v1), encoder(v2))
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        total += loss.item()
    print(f"[Pre-train] epoch {ep}: loss = {total/len(pre_dl):.4f}")

# ---------- fine-tuning ---------------------------------------
n_classes = len(set(r["label"] for r in real_dset["train"]))
clf = Classifier(encoder, n_classes).to(device)
opt_ft = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
FT_EPOCHS = 10


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, acc_loss = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids, lbl = batch["ids"].to(device), batch["label"].to(device)
            logits = model(ids)
            acc_loss += criterion(logits, lbl).item()
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(lbl.cpu().tolist())
            seqs.extend(batch["seq"])
    return acc_loss / len(loader), ccwa_metric(seqs, gts, preds), preds, gts


for ep in range(1, FT_EPOCHS + 1):
    clf.train()
    tloss, steps = 0.0, 0
    for batch in train_dl:
        ids, lbl = batch["ids"].to(device), batch["label"].to(device)
        loss = criterion(clf(ids), lbl)
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        tloss += loss.item()
        steps += 1
    tloss /= steps

    vloss, vccwa, vpred, vgt = evaluate(clf, dev_dl)
    ts = datetime.datetime.now().isoformat()

    ed = experiment_data["mask_only"]["SPR_BENCH"]
    ed["losses"]["train"].append(tloss)
    ed["losses"]["val"].append(vloss)
    ed["metrics"]["train_CCWA"].append(None)
    ed["metrics"]["val_CCWA"].append(vccwa)
    ed["predictions"].append(vpred)
    ed["ground_truth"].append(vgt)
    ed["timestamps"].append(ts)

    print(
        f"[FT] epoch {ep}: train_loss={tloss:.4f} | val_loss={vloss:.4f} | CCWA={vccwa:.4f}"
    )

# ---------- save results --------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved metrics to {os.path.join(working_dir, 'experiment_data.npy')}")
