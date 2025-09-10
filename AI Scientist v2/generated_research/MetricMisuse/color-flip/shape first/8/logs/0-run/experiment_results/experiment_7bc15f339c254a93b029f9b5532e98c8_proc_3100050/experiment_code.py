import os, random, pathlib, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --------------------------------------------------------------------------- #
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------------- #
# ---------- try to load the real SPR_BENCH (falls back to synthetic) --------
def try_load_spr():
    try:
        from SPR import load_spr_bench

        root = pathlib.Path("./SPR_BENCH")
        if not root.exists():
            root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        ds = load_spr_bench(root)
        # convert HF Dataset rows to list[dict] for easy indexing
        return {
            k: [dict(ds[k][i]) for i in range(len(ds[k]))]
            for k in ["train", "dev", "test"]
        }
    except Exception as e:
        print("Falling back to synthetic data =>", e)
        return None


real_ds = try_load_spr()

# ------------------ synthetic fallback (easy but small) ---------------------
SHAPES, COLORS = list("RSTUVW"), list("ABCDE")


def rand_token():
    return random.choice(SHAPES) + random.choice(COLORS)


def make_row(i):
    return {
        "id": i,
        "sequence": " ".join(rand_token() for _ in range(random.randint(3, 10))),
        "label": random.randint(0, 3),
    }


if real_ds is None:
    real_ds = {
        "train": [make_row(i) for i in range(4000)],
        "dev": [make_row(4000 + i) for i in range(800)],
        "test": [make_row(4800 + i) for i in range(800)],
    }

# ----------------------- helpers / vocabulary --------------------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    vocab = set()
    [vocab.update(r["sequence"].split()) for r in rows]
    lst = [PAD, UNK, MASK] + sorted(vocab)
    return {t: i for i, t in enumerate(lst)}, lst


stoi, itos = build_vocab(real_ds["train"])
vocab_size = len(stoi)


def tokens_to_ids(tok_lst):
    return [stoi.get(t, stoi[UNK]) for t in tok_lst]


# -------------------------- data augmentation -------------------------------
def augment_tokens(tok_lst):
    toks = tok_lst[:]  # shallow copy
    if random.random() < 0.3 and len(toks) > 1:  # random swap
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    toks = [MASK if random.random() < 0.15 else t for t in toks]  # random mask
    return toks


# ------------------------------ datasets ------------------------------------
class ContrastiveDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq_str = self.rows[idx]["sequence"]
        toks = seq_str.split()
        view1 = tokens_to_ids(augment_tokens(toks))
        view2 = tokens_to_ids(augment_tokens(toks))
        return {
            "v1": torch.tensor(view1, dtype=torch.long),
            "v2": torch.tensor(view2, dtype=torch.long),
        }


class SupervisedDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = tokens_to_ids(r["sequence"].split())
        lab = int(r["label"])
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(lab, dtype=torch.long),
            "seq": r["sequence"],
        }


# ------------------ custom collators (dynamic padding) ----------------------
def pad_collate_contrastive(batch):
    v1 = pad_sequence(
        [b["v1"] for b in batch], batch_first=True, padding_value=stoi[PAD]
    )
    v2 = pad_sequence(
        [b["v2"] for b in batch], batch_first=True, padding_value=stoi[PAD]
    )
    return {"view1": v1.to(device), "view2": v2.to(device)}


def pad_collate_supervised(batch):
    ids = pad_sequence(
        [b["ids"] for b in batch], batch_first=True, padding_value=stoi[PAD]
    ).to(device)
    labels = torch.stack([b["label"] for b in batch]).to(device)
    seqs = [b["seq"] for b in batch]
    return {"ids": ids, "label": labels, "seq": seqs}


# -------------------------- metric utilities --------------------------------
def count_shape(seq):
    return len({tok[0] for tok in seq.split()})


def count_color(seq):
    return len({tok[1] for tok in seq.split()})


def ccwa_metric(seqs, y_true, y_pred):
    w = [count_shape(s) + count_color(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ------------------------------ model ---------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, proj_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=stoi[PAD])
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hid * 2, proj_dim)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        z = nn.functional.normalize(self.proj(h), dim=1)
        return z


class Classifier(nn.Module):
    def __init__(self, encoder, num_cls):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(128, num_cls)

    def forward(self, x):
        return self.head(self.enc(x))


def simclr_loss(z1, z2, temp=0.5):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], 0)
    logits = (reps @ reps.T) / temp
    logits.fill_diagonal_(-9e15)
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(device)
    return nn.functional.cross_entropy(logits, pos)


# --------------------------- data loaders -----------------------------------
BATCH = 256
train_unlab_dl = DataLoader(
    ContrastiveDS(real_ds["train"]),
    batch_size=BATCH,
    shuffle=True,
    drop_last=True,
    collate_fn=pad_collate_contrastive,
)
train_sup_dl = DataLoader(
    SupervisedDS(real_ds["train"]),
    batch_size=BATCH,
    shuffle=True,
    collate_fn=pad_collate_supervised,
)
dev_dl = DataLoader(
    SupervisedDS(real_ds["dev"]), batch_size=BATCH, collate_fn=pad_collate_supervised
)

# ----------------------------- experiment log -------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_CCWA": [], "val_CCWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------------ pre-training (contrastive) ------------------------
num_classes = len({r["label"] for r in real_ds["train"]})
encoder = Encoder(vocab_size).to(device)
opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
pre_epochs = 3
for ep in range(1, pre_epochs + 1):
    encoder.train()
    tot = 0
    n = 0
    for batch in train_unlab_dl:
        loss = simclr_loss(encoder(batch["view1"]), encoder(batch["view2"]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
    print(f"Pre-Epoch {ep}: contrastive_loss = {tot/n:.4f}")

# --------------------------- fine-tuning phase ------------------------------
clf = Classifier(encoder, num_classes).to(device)
ft_opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
ft_epochs = 4


def evaluate(model, dl):
    model.eval()
    tot = 0
    n = 0
    preds = []
    ys = []
    seqs = []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["ids"])
            loss = criterion(out, batch["label"])
            tot += loss.item()
            n += 1
            preds += out.argmax(1).cpu().tolist()
            ys += batch["label"].cpu().tolist()
            seqs += batch["seq"]
    return tot / n, ccwa_metric(seqs, ys, preds), preds, ys, seqs


for ep in range(1, ft_epochs + 1):
    clf.train()
    tot = 0
    n = 0
    for batch in train_sup_dl:
        out = clf(batch["ids"])
        loss = criterion(out, batch["label"])
        ft_opt.zero_grad()
        loss.backward()
        ft_opt.step()
        tot += loss.item()
        n += 1
    train_loss = tot / n
    val_loss, val_ccwa, preds, gt, seqs = evaluate(clf, dev_dl)
    ts = datetime.datetime.now().isoformat()
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_CCWA"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val_CCWA"].append(val_ccwa)
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(gt)
    experiment_data["SPR_BENCH"]["timestamps"].append(ts)
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f} | CCWA = {val_ccwa:.4f}")

# ------------------------------ save results --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
