import os, random, math, pathlib, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ---------------------------------------------------------------------
# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# ---------------------- Data utilities --------------------------------
def load_spr_bench(root: pathlib.Path):
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
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print(f"Found SPR_BENCH at {p}")
            return load_spr_bench(p)
    return None


dset = try_load_spr()

# ---------------- synthetic fallback if no data ------------------------
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


if dset is None:
    print("Real SPR_BENCH not found, using synthetic data.")
    dset = {
        "train": make_split(6000),
        "dev": make_split(1200),
        "test": make_split(1200),
    }

# ---------------------------------------------------------------------
# ------------- Complexity-Adjusted Weighted Accuracy ------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def cawa_metric(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ---------------------------------------------------------------------
# ------------------------ vocabulary ----------------------------------
def build_vocab(rows):
    vocab = set()
    for r in rows:
        vocab.update(r["sequence"].split())
    itos = [PAD, UNK, MASK] + sorted(vocab)
    return {t: i for i, t in enumerate(itos)}, itos


stoi, itos = build_vocab(dset["train"])
vocab_size, MAX_LEN = len(stoi), 20


def encode(seq: str):
    ids = [stoi.get(tok, stoi[UNK]) for tok in seq.split()][:MAX_LEN]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


# ---------------------------------------------------------------------
# ----------------------- datasets  ------------------------------------
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


# ---------------------------------------------------------------------
# -------------------- model definitions ------------------------------
class TransEncoderNoPos(nn.Module):
    def __init__(self, vocab, d_model=96, nhead=6, nlayers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_model, 128)

    def forward(self, x):
        h = self.transformer(self.emb(x.to(device)))
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, n_cls):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------------------------------------------------------------------
# ---------------------- SimCLR loss ----------------------------------
def simclr_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.T / temperature
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), -9e15)
    positives = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, positives)


# ---------------------------------------------------------------------
# --------------------- experiment bookkeeping ------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_CAWA": [], "val_CAWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------------------------------------------------------------
# --------------------------- DataLoaders -----------------------------
BATCH_PRE, BATCH_FT = 256, 256
pre_dl = DataLoader(
    ContrastiveSPR(dset["train"]), batch_size=BATCH_PRE, shuffle=True, drop_last=True
)
train_dl = DataLoader(SupervisedSPR(dset["train"]), batch_size=BATCH_FT, shuffle=True)
dev_dl = DataLoader(SupervisedSPR(dset["dev"]), batch_size=BATCH_FT, shuffle=False)

# ---------------------------------------------------------------------
# --------------------------- pre-training ----------------------------
encoder = TransEncoderNoPos(vocab_size).to(device)
opt_pre = torch.optim.Adam(encoder.parameters(), lr=1e-3)
PRE_EPOCHS = 6
for ep in range(1, PRE_EPOCHS + 1):
    encoder.train()
    run = 0.0
    for batch in pre_dl:
        v1, v2 = batch["view1"].to(device), batch["view2"].to(device)
        loss = simclr_loss(encoder(v1), encoder(v2))
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        run += loss.item()
    print(f"Pre-train epoch {ep}: contrastive_loss={run/len(pre_dl):.4f}")

# ---------------------------------------------------------------------
# --------------------------- fine-tuning -----------------------------
n_classes = len(set(r["label"] for r in dset["train"]))
clf = Classifier(encoder, n_classes).to(device)

# freeze encoder for warm-up
for p in clf.encoder.parameters():
    p.requires_grad = False
opt_ft = torch.optim.Adam(
    [
        {"params": clf.fc.parameters(), "lr": 2e-3},
        {"params": clf.encoder.parameters(), "lr": 5e-4},
    ]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_ft, mode="min", factor=0.5, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss()
FT_EPOCHS, WARM_EPOCHS, patience, best_val = 20, 2, 4, float("inf")
patience_ctr = 0


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, loss_acc = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["ids"])
            loss = criterion(logits, batch["label"])
            loss_acc += loss.item()
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    return (loss_acc / len(loader), cawa_metric(seqs, gts, preds), preds, gts)


for ep in range(1, FT_EPOCHS + 1):
    # unfreeze after warm-up
    if ep == WARM_EPOCHS + 1:
        for p in clf.encoder.parameters():
            p.requires_grad = True

    clf.train()
    tr_loss, tr_preds, tr_gts, tr_seqs = 0.0, [], [], []
    for batch in train_dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = clf(batch["ids"])
        loss = criterion(logits, batch["label"])
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        tr_loss += loss.item()
        tr_preds.extend(logits.argmax(1).cpu().tolist())
        tr_gts.extend(batch["label"].cpu().tolist())
        tr_seqs.extend(batch["seq"])
    tr_loss /= len(train_dl)
    train_cawa = cawa_metric(tr_seqs, tr_gts, tr_preds)

    val_loss, val_cawa, val_pred, val_gt = evaluate(clf, dev_dl)
    scheduler.step(val_loss)

    ts = datetime.datetime.now().isoformat()
    exp = experiment_data["SPR_BENCH"]
    exp["losses"]["train"].append(tr_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["train_CAWA"].append(train_cawa)
    exp["metrics"]["val_CAWA"].append(val_cawa)
    exp["predictions"].append(val_pred)
    exp["ground_truth"].append(val_gt)
    exp["timestamps"].append(ts)

    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
        f"train_CAWA={train_cawa:.4f} | val_CAWA={val_cawa:.4f}"
    )

    # -------- early stopping ------------
    if val_loss < best_val - 1e-4:
        best_val, patience_ctr = val_loss, 0
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print("Early stopping triggered.")
            break

# ---------------------------------------------------------------------
# --------------------------- save results ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
