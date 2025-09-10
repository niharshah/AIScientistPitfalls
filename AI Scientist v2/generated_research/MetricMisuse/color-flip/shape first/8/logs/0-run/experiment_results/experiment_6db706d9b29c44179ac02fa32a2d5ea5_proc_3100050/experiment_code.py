import os, random, datetime, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------- workspace ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset loading ----
def load_or_make_dataset():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        if not DATA_PATH.exists():
            DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH")
        return {k: [dset[k][i] for i in range(len(dset[k]))] for k in dset}
    except Exception as e:
        print("Falling back to synthetic data:", e)

    shapes, colors = list("RSTUV"), list("ABCDE")

    def rand_token():
        return random.choice(shapes) + random.choice(colors)

    def rand_seq():
        return " ".join(rand_token() for _ in range(random.randint(3, 10)))

    def make_split(n):
        return [
            {"id": i, "sequence": rand_seq(), "label": random.randint(0, 3)}
            for i in range(n)
        ]

    return {"train": make_split(1000), "dev": make_split(200), "test": make_split(200)}


data = load_or_make_dataset()

# ---------- vocab --------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    tokset = set()
    for r in rows:
        tokset.update(r["sequence"].split())
    itos = [PAD, UNK, MASK] + sorted(tokset)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


stoi, itos = build_vocab(data["train"])
vocab_size = len(stoi)
MAX_LEN = 20


def encode(seq):
    ids = [stoi.get(t, stoi[UNK]) for t in seq.split()][:MAX_LEN]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


# ---------- metrics ------------
def count_shape(seq):
    return len({tok[0] for tok in seq.split()})


def count_color(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def ccwa(seqs, y_t, y_p):
    w = [count_shape(s) + count_color(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(cor) / sum(w) if sum(w) else 0.0


# ---------- datasets ----------
class ContrastiveSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def _aug(self, ids):
        tok = [t for t in ids if t != stoi[PAD]]
        if not tok:
            tok = [stoi[PAD]]
        if random.random() < 0.3:
            tok.pop(random.randrange(len(tok)))  # delete
        if len(tok) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(tok)), 2)
            tok[i], tok[j] = tok[j], tok[i]  # permute
        tok = [stoi[MASK] if random.random() < 0.15 else t for t in tok]  # mask
        return encode(" ".join(itos[t] for t in tok))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = encode(self.rows[idx]["sequence"])
        return {"v1": torch.tensor(self._aug(ids)), "v2": torch.tensor(self._aug(ids))}


class SupervisedSPR(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "ids": torch.tensor(encode(r["sequence"])),
            "label": torch.tensor(r["label"]),
            "seq": r["sequence"],
        }


# ---------- model --------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, 128)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, enc, num_classes):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.enc(x))


# ---------- losses -------------
def simclr(z1, z2, temp=0.5):
    z1, z2 = [nn.functional.normalize(z, dim=1) for z in (z1, z2)]
    N = z1.size(0)
    reps = torch.cat([z1, z2], 0)
    logits = reps @ reps.t() / temp
    logits.fill_diagonal_(-1e9)
    pos_idx = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(device)
    return nn.functional.cross_entropy(logits, pos_idx)


# ---------- helpers ------------
def evaluate(model, loader, crit):
    model.eval()
    tot, cnt = 0, 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch["ids"])
            loss = crit(out, batch["label"])
            tot += loss.item()
            cnt += 1
            preds += out.argmax(1).cpu().tolist()
            labels += batch["label"].cpu().tolist()
            seqs += batch["seq"]
    return tot / cnt if cnt else 0.0, ccwa(seqs, labels, preds), preds, labels, seqs


# ---------- experiment dict ----
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"val_CCWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------- training -----------
BATCH = 256
num_cls = len(set(r["label"] for r in data["train"]))
contrast_ds = ContrastiveSPR(data["train"])
contr_dl = DataLoader(contrast_ds, batch_size=BATCH, shuffle=True, drop_last=True)

enc = Encoder().to(device)
opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
PRE_EPOCHS = 5
for ep in range(1, PRE_EPOCHS + 1):
    enc.train()
    tot = 0
    n = 0
    for batch in contr_dl:
        v1 = batch["v1"].to(device)
        v2 = batch["v2"].to(device)
        loss = simclr(enc(v1), enc(v2))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
    print(f"Pretrain Epoch {ep}: loss={tot/n:.4f}")

# ---------- fine-tune ----------
train_ds = SupervisedSPR(data["train"])
dev_ds = SupervisedSPR(data["dev"])
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=BATCH, shuffle=False)
clf = Classifier(enc, num_cls).to(device)
ft_opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
FT_EPOCHS = 10
for ep in range(1, FT_EPOCHS + 1):
    clf.train()
    tot = 0
    n = 0
    for batch in train_dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        loss = criterion(clf(batch["ids"]), batch["label"])
        ft_opt.zero_grad()
        loss.backward()
        ft_opt.step()
        tot += loss.item()
        n += 1
    train_loss = tot / n
    val_loss, val_ccwa, preds, gt, seqs = evaluate(clf, dev_dl, criterion)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f} | CCWA = {val_ccwa:.4f}")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_CCWA"].append(val_ccwa)
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(gt)
    experiment_data["SPR_BENCH"]["timestamps"].append(ts)

# ---------- save ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
