# ---------------- BiLSTM encoder ablation study -----------------
import os, random, math, pathlib, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------- directories / device ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- optional SPR benchmark ----------------------
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
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print(f"Found SPR_BENCH at {p}")
            return load_spr_bench(p)
    return None


real_dset = try_load_spr()

# -------------- synthetic fallback if benchmark missing --------
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


# ------------------- CCWA metric helpers -----------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def ccwa_metric(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ------------------------ vocabulary ---------------------------
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


# ------------------------- datasets ----------------------------
class ContrastiveSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def _augment(self, ids):
        toks = [t for t in ids if t != stoi[PAD]]
        if random.random() < 0.3 and len(toks) > 1:
            toks.pop(random.randrange(len(toks)))  # deletion
        if random.random() < 0.3 and len(toks) > 2:  # swap
            i = random.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        if random.random() < 0.3:
            toks += random.sample(toks, k=1)  # dup
        toks = [stoi[MASK] if random.random() < 0.15 else t for t in toks]  # mask
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


# ----------------------- BiLSTM encoder ------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab, d_model=96, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        hidden = d_model // 2
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=nlayers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(d_model, 128)

    def forward(self, x_ids):
        mask = (x_ids != stoi[PAD]).to(device)
        x = self.emb(x_ids.to(device))
        h, _ = self.lstm(x)  # h: B x T x d_model
        # length-masked mean pooling
        h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------------------- SimCLR / InfoNCE -----------------------
def simclr_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    z = nn.functional.normalize(z, dim=1)
    sim = (z @ z.T) / temperature
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), -1e9)
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, pos_idx)


# -------------------- dataloaders ------------------------------
BATCH_PRE, BATCH_FT = 256, 256
pre_ds = ContrastiveSPR(real_dset["train"])
train_ds, dev_ds = SupervisedSPR(real_dset["train"]), SupervisedSPR(real_dset["dev"])
pre_dl = DataLoader(pre_ds, batch_size=BATCH_PRE, shuffle=True, drop_last=True)
train_dl = DataLoader(train_ds, batch_size=BATCH_FT, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=BATCH_FT, shuffle=False)

# ------------------ experiment bookkeeping --------------------
experiment_data = {
    "BiLSTM": {
        "SPR_BENCH": {
            "metrics": {"train_CCWA": [], "val_CCWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# -------------------- contrastive pre-training -----------------
encoder = BiLSTMEncoder(vocab_size).to(device)
opt_pre = torch.optim.Adam(encoder.parameters(), lr=1e-3)
for ep in range(1, 7):
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
    print(f"Pre-train epoch {ep}: contrastive_loss={running/len(pre_dl):.4f}")

# ---------------------- supervised fine-tune -------------------
n_classes = len({r["label"] for r in real_dset["train"]})
clf = Classifier(encoder, n_classes).to(device)
opt_ft = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, gts, seqs, loss_sum = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids, label = batch["ids"].to(device), batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, label)
            loss_sum += loss.item()
            preds += logits.argmax(1).cpu().tolist()
            gts += label.cpu().tolist()
            seqs += batch["seq"]
    return loss_sum / len(loader), ccwa_metric(seqs, gts, preds), preds, gts


for ep in range(1, 11):
    clf.train()
    train_loss, steps = 0.0, 0
    for batch in train_dl:
        ids, label = batch["ids"].to(device), batch["label"].to(device)
        loss = criterion(clf(ids), label)
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        train_loss += loss.item()
        steps += 1
    train_loss /= steps
    val_loss, val_ccwa, val_preds, val_gt = evaluate(clf, dev_dl)
    ts = datetime.datetime.now().isoformat()
    ed = experiment_data["BiLSTM"]["SPR_BENCH"]
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

# -------------------------- save --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
