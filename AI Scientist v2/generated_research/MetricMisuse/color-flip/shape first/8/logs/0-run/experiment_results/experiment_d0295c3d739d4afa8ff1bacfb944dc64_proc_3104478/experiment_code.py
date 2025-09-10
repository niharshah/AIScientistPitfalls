# -------------------------------------------------------------
# No-Pretraining (Supervised-Only) Ablation
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


# ---------- helper : load real benchmark if available --------
def load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}


def try_load_spr():
    for root in (
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ):
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
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def ccwa_metric(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------- vocabulary ---------------------------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    vocab = set()
    for r in rows:
        vocab.update(r["sequence"].split())
    itos = [PAD, UNK, MASK] + sorted(vocab)
    return {t: i for i, t in enumerate(itos)}, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size, MAX_LEN = len(stoi), 20


def encode(seq: str):
    ids = [stoi.get(tok, stoi[UNK]) for tok in seq.split()][:MAX_LEN]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


# ---------- datasets -----------------------------------------
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


# ---------- model --------------------------------------------
class TransEncoder(nn.Module):
    def __init__(self, vocab, d_model=96, nhead=6, nlayers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(MAX_LEN, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, nlayers)
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


# ---------- loaders ------------------------------------------
BATCH = 256
train_ds = SupervisedSPR(real_dset["train"])
dev_ds = SupervisedSPR(real_dset["dev"])
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=BATCH, shuffle=False)

# ---------- experiment bookkeeping ---------------------------
experiment_data = {
    "no_pretrain": {
        "SPR_BENCH": {
            "metrics": {"train_CCWA": [], "val_CCWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ---------- supervised training ------------------------------
encoder = TransEncoder(vocab_size).to(device)  # random init, no pretrain
n_classes = len(set(r["label"] for r in real_dset["train"]))
clf = Classifier(encoder, n_classes).to(device)
opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS = 10


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, loss_tot = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["ids"])
            loss = criterion(logits, batch["label"])
            loss_tot += loss.item()
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    return loss_tot / len(loader), ccwa_metric(seqs, gts, preds), preds, gts


for ep in range(1, EPOCHS + 1):
    clf.train()
    tr_loss, steps = 0.0, 0
    for batch in train_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        loss = criterion(clf(batch["ids"]), batch["label"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tr_loss += loss.item()
        steps += 1
    tr_loss /= steps

    val_loss, val_ccwa, val_preds, val_gt = evaluate(clf, dev_dl)
    timestamp = datetime.datetime.now().isoformat()

    # log
    bench = experiment_data["no_pretrain"]["SPR_BENCH"]
    bench["losses"]["train"].append(tr_loss)
    bench["losses"]["val"].append(val_loss)
    bench["metrics"]["train_CCWA"].append(None)
    bench["metrics"]["val_CCWA"].append(val_ccwa)
    bench["predictions"].append(val_preds)
    bench["ground_truth"].append(val_gt)
    bench["timestamps"].append(timestamp)

    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | CCWA={val_ccwa:.4f}"
    )

# ---------- save ---------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved metrics to {os.path.join(working_dir,'experiment_data.npy')}")
