# -------------------------------------------------------------
# No-Augmentation-Contrastive ablation – complete experiment run
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


# ---------- utility to (try to) load the real benchmark -------
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
    for root in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
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


# ---------- CCWA metric helpers -------------------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def ccwa_metric(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


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


def encode(seq: str):
    ids = [stoi.get(tok, stoi[UNK]) for tok in seq.split()][:MAX_LEN]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


# ---------- datasets ------------------------------------------
class ContrastiveSPR(Dataset):
    """No-augmentation version: both views are identical."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.rows[idx]["sequence"]), dtype=torch.long)
        return {"view1": ids, "view2": ids.clone()}


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
            d_model, nhead, 4 * d_model, batch_first=True
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


# ---------- SimCLR / InfoNCE loss ------------------------------
def simclr_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, pos_idx)


# ---------- experiment bookkeeping -----------------------------
experiment_data = {
    "NoAugContrastive": {
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
opt_pre, PRE_EPOCHS = torch.optim.Adam(encoder.parameters(), 1e-3), 6
for ep in range(1, PRE_EPOCHS + 1):
    encoder.train()
    running = 0.0
    for batch in pre_dl:
        loss = simclr_loss(
            encoder(batch["view1"].to(device)), encoder(batch["view2"].to(device))
        )
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        running += loss.item()
    print(f"Pre-train epoch {ep}: contrastive_loss = {running/len(pre_dl):.4f}")

# ---------- fine-tuning ---------------------------------------
n_classes = len({r["label"] for r in real_dset["train"]})
clf, opt_ft = Classifier(encoder, n_classes).to(device), torch.optim.Adam(
    encoder.parameters(), 2e-3
)
criterion, FT_EPOCHS = nn.CrossEntropyLoss(), 10


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, loss_sum = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch_tensor = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch_tensor["ids"])
            loss = criterion(logits, batch_tensor["label"])
            loss_sum += loss.item()
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch_tensor["label"].cpu().tolist())
            seqs.extend(batch_tensor["seq"])
    return loss_sum / len(loader), ccwa_metric(seqs, gts, preds), preds, gts


for ep in range(1, FT_EPOCHS + 1):
    clf.train()
    train_loss, steps = 0.0, 0
    for batch in train_dl:
        bt = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        loss = criterion(clf(bt["ids"]), bt["label"])
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        train_loss += loss.item()
        steps += 1
    train_loss /= steps

    val_loss, val_ccwa, val_preds, val_gt = evaluate(clf, dev_dl)
    ts = datetime.datetime.now().isoformat()

    ed = experiment_data["NoAugContrastive"]["SPR_BENCH"]
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

# ---------- save --------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved all metrics to {os.path.join(working_dir, 'experiment_data.npy')}")
