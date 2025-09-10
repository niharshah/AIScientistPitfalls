# -------------------------------------------------------------
# Context-aware contrastive learning – ablation: NO PROJECTION
# -------------------------------------------------------------
import os, random, math, pathlib, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- mandatory working directory ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device -------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper to load real SPR_BENCH --------------------
def load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


def try_load_spr():
    for p in (
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ):
        if (p / "train.csv").exists():
            print(f"Found SPR_BENCH at {p}")
            return load_spr_bench(p)
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
    return len({t[0] for t in seq.split() if t})


def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def ccwa_metric(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# ---------- vocabulary ----------------------------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    vocab = set()
    for r in rows:
        vocab.update(r["sequence"].split())
    itos = [PAD, UNK, MASK] + sorted(vocab)
    return {t: i for i, t in enumerate(itos)}, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size, MAX_LEN = len(stoi), 20


def encode(seq):
    ids = [stoi.get(tok, stoi[UNK]) for tok in seq.split()][:MAX_LEN]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


# ---------- datasets ------------------------------------------
class ContrastiveSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def _augment(self, ids):
        toks = [t for t in ids if t != stoi[PAD]]
        if random.random() < 0.3 and len(toks) > 1:
            toks.pop(random.randrange(len(toks)))
        if random.random() < 0.3 and len(toks) > 2:
            i = random.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        if random.random() < 0.3:
            toks += random.sample(toks, k=1)
        toks = [stoi[MASK] if random.random() < 0.15 else t for t in toks]
        return torch.tensor(encode(" ".join(itos[t] for t in toks)), dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        base = torch.tensor(encode(self.rows[i]["sequence"]), dtype=torch.long)
        return {"view1": self._augment(base), "view2": self._augment(base)}


class SupervisedSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "ids": torch.tensor(encode(r["sequence"]), dtype=torch.long),
            "label": torch.tensor(r["label"], dtype=torch.long),
            "seq": r["sequence"],
        }


# ---------- model (projection removed) ------------------------
class TransEncoder(nn.Module):
    def __init__(self, vocab, d_model=96, nhead=6, nlayers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.emb(x.to(device)) + self.pos[: x.size(1)].unsqueeze(0)
        h = self.transformer(x)  # B x L x d
        h = self.pool(h.transpose(1, 2)).squeeze(-1)  # B x d
        return h  # dimension 96


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(96, n_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------- SimCLR loss ---------------------------------------
def simclr_loss(z1, z2, temp=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim = (z @ z.T) / temp
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    pos = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, pos)


# ---------- experiment bookkeeping ----------------------------
experiment_data = {
    "no_proj_head": {
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
pre_dl = DataLoader(
    ContrastiveSPR(real_dset["train"]),
    batch_size=BATCH_PRE,
    shuffle=True,
    drop_last=True,
)
train_dl = DataLoader(
    SupervisedSPR(real_dset["train"]), batch_size=BATCH_FT, shuffle=True
)
dev_dl = DataLoader(SupervisedSPR(real_dset["dev"]), batch_size=BATCH_FT, shuffle=False)

# ---------- training ------------------------------------------
encoder = TransEncoder(vocab_size).to(device)
opt_pre = torch.optim.Adam(encoder.parameters(), lr=1e-3)
for ep in range(1, 7):
    encoder.train()
    run = 0.0
    for b in pre_dl:
        v1, v2 = b["view1"].to(device), b["view2"].to(device)
        loss = simclr_loss(encoder(v1), encoder(v2))
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        run += loss.item()
    print(f"Pre-train epoch {ep}: contrastive_loss={run/len(pre_dl):.4f}")

# ---------- fine-tuning ---------------------------------------
n_classes = len(set(r["label"] for r in real_dset["train"]))
clf = Classifier(encoder, n_classes).to(device)
opt_ft = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, loss_total = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss_total += loss.item()
            preds += logits.argmax(1).cpu().tolist()
            gts += labels.cpu().tolist()
            seqs += batch["seq"]
    return loss_total / len(loader), ccwa_metric(seqs, gts, preds), preds, gts


for ep in range(1, 11):
    clf.train()
    train_loss, steps = 0.0, 0
    for batch in train_dl:
        ids = batch["ids"].to(device)
        labels = batch["label"].to(device)
        loss = criterion(clf(ids), labels)
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        train_loss += loss.item()
        steps += 1
    train_loss /= steps
    val_loss, val_ccwa, val_pred, val_gt = evaluate(clf, dev_dl)
    ts = datetime.datetime.now().isoformat()

    e = experiment_data["no_proj_head"]["SPR_BENCH"]
    e["losses"]["train"].append(train_loss)
    e["losses"]["val"].append(val_loss)
    e["metrics"]["train_CCWA"].append(None)
    e["metrics"]["val_CCWA"].append(val_ccwa)
    e["predictions"].append(val_pred)
    e["ground_truth"].append(val_gt)
    e["timestamps"].append(ts)

    print(
        f"Epoch {ep}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CCWA={val_ccwa:.4f}"
    )

# ---------- save ------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved metrics to {os.path.join(working_dir,'experiment_data.npy')}")
