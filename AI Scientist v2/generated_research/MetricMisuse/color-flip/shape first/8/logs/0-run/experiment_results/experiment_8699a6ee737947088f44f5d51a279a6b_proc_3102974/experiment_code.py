# -------------------------------------------------------------
# Context-aware contrastive learning – ablation: no positional embeddings
# -------------------------------------------------------------
import os, random, math, pathlib, datetime, numpy as np, torch, sys
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- working directory --------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device -------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper to load SPR benchmark ---------------------
def load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess

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
    for path in ["./SPR_BENCH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH"]:
        root = pathlib.Path(path)
        if (root / "train.csv").exists():
            print(f"Found SPR_BENCH at {root}")
            return load_spr_bench(root)
    return None


real_dset = try_load_spr()

# ---------- synthetic fallback --------------------------------
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


# ---------- CCWA metric --------------------------------------
def count_shape(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def ccwa_metric(seqs, y_true, y_pred):
    w = [count_shape(s) + count_color(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ---------- vocabulary ----------------------------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    vocab = set()
    for r in rows:
        vocab.update(r["sequence"].split())
    itos = [PAD, UNK, MASK] + sorted(vocab)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


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

    def __getitem__(self, idx):
        base = torch.tensor(encode(self.rows[idx]["sequence"]), dtype=torch.long)
        return {"view1": self._augment(base), "view2": self._augment(base)}


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


# ---------- model (positional embeddings removed) -------------
class TransEncoder(nn.Module):
    def __init__(self, vocab, d_model=96, nhead=6, nlayers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        # NOTE: no learnable positional encoding
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_model, 128)

    def forward(self, x):
        x = self.emb(x.to(device))  # (B,L,D)
        h = self.transformer(x)  # (B,L,D)
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------- SimCLR / InfoNCE loss -----------------------------
def simclr_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), -1e9)
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, pos_idx)


# ---------- bookkeeping dict ----------------------------------
experiment_data = {
    "remove_positional_embeddings": {
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
BATCH_PRE = 256
BATCH_FT = 256
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
    print(f"Pre-train epoch {ep}: contrastive_loss={running/len(pre_dl):.4f}")

# ---------- fine-tuning ---------------------------------------
n_classes = len(set(r["label"] for r in real_dset["train"]))
clf = Classifier(encoder, n_classes).to(device)
opt_ft = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, tot_loss = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["ids"])
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    return tot_loss / len(loader), ccwa_metric(seqs, gts, preds), preds, gts


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
    timestamp = datetime.datetime.now().isoformat()

    ed = experiment_data["remove_positional_embeddings"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_CCWA"].append(None)
    ed["metrics"]["val_CCWA"].append(val_ccwa)
    ed["predictions"].append(val_preds)
    ed["ground_truth"].append(val_gt)
    ed["timestamps"].append(timestamp)

    print(
        f"Epoch {ep}: train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | CCWA={val_ccwa:.4f}"
    )

# ---------- save results --------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved all metrics to {os.path.join(working_dir,'experiment_data.npy')}")
