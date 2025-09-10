# -------------------------------------------------------------
# CLS-POOL ONLY ablation – single-file runnable script
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


# ---------- dataset loader helper ----------------------------
def load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    def _load(csv_name: str):
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
    candidate_roots = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for root in candidate_roots:
        if (root / "train.csv").exists():
            print(f"Found SPR_BENCH at {root}")
            return load_spr_bench(root)
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
    weights = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ---------- vocabulary & encoding -----------------------------
PAD, UNK, MASK, CLS = "<PAD>", "<UNK>", "<MASK>", "<CLS>"


def build_vocab(rows):
    vocab = set()
    for r in rows:
        vocab.update(r["sequence"].split())
    itos = [PAD, UNK, MASK, CLS] + sorted(vocab)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size, MAX_LEN = len(stoi), 20  # includes CLS position


def encode(seq: str):
    toks = [CLS] + seq.split()  # prepend CLS
    ids = [stoi.get(t, stoi[UNK]) for t in toks][:MAX_LEN]  # truncate
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))  # pad
    return ids


# ---------- datasets ------------------------------------------
class ContrastiveSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def _augment(self, ids_tensor):
        ids = ids_tensor.tolist()
        toks = [itos[t] for t in ids if t not in (stoi[PAD], stoi[CLS])]
        # deletion
        if random.random() < 0.3 and len(toks) > 1:
            toks.pop(random.randrange(len(toks)))
        # swap
        if random.random() < 0.3 and len(toks) > 2:
            i = random.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        # duplication
        if random.random() < 0.3 and toks:
            toks += random.sample(toks, k=1)
        # random mask
        toks = [MASK if random.random() < 0.15 else t for t in toks]
        aug_seq = " ".join(toks)
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
        self.proj = nn.Linear(d_model, 128)

    def forward(self, x):
        x = self.emb(x.to(device)) + self.pos[: x.size(1)].unsqueeze(0)
        h = self.transformer(x)
        cls_emb = h[:, 0, :]  # CLS-POOL ONLY
        return torch.tanh(self.proj(cls_emb))


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
    z = torch.cat([z1, z2], 0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * B, device=z.device, dtype=torch.bool), -1e9)
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, pos_idx)


# ---------- bookkeeping dict ----------------------------------
experiment_data = {
    "CLS_POOL_ONLY": {
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

# ---------- pre-training --------------------------------------
encoder = TransEncoder(vocab_size).to(device)
opt_pre = torch.optim.Adam(encoder.parameters(), lr=1e-3)
for ep in range(1, 7):
    encoder.train()
    running = 0.0
    for batch in pre_dl:
        v1, v2 = batch["view1"].to(device), batch["view2"].to(device)
        loss = simclr_loss(encoder(v1), encoder(v2))
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        running += loss.item()
    print(f"Pre-train epoch {ep}: contrastive_loss = {running/len(pre_dl):.4f}")

# ---------- fine-tuning ---------------------------------------
n_classes = len({r["label"] for r in real_dset["train"]})
clf = Classifier(encoder, n_classes).to(device)
opt_ft = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, loss_accum = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["ids"])
            loss_accum += criterion(logits, batch["label"]).item()
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    return loss_accum / len(loader), ccwa_metric(seqs, gts, preds), preds, gts


for ep in range(1, 11):
    clf.train()
    train_loss, steps = 0.0, 0
    for batch in train_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        loss = criterion(clf(batch["ids"]), batch["label"])
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        train_loss += loss.item()
        steps += 1
    train_loss /= steps

    val_loss, val_ccwa, val_preds, val_gt = evaluate(clf, dev_dl)
    ts = datetime.datetime.now().isoformat()
    ed = experiment_data["CLS_POOL_ONLY"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_CCWA"].append(None)
    ed["metrics"]["val_CCWA"].append(val_ccwa)
    ed["predictions"].append(val_preds)
    ed["ground_truth"].append(val_gt)
    ed["timestamps"].append(ts)
    print(
        f"Epoch {ep}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CCWA={val_ccwa:.4f}"
    )

# ---------- save ----------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved all metrics to {os.path.join(working_dir, 'experiment_data.npy')}")
