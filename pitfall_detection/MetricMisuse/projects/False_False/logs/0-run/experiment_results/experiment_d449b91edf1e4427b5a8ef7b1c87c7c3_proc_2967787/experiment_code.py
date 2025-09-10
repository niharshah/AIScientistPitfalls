import os, random, time, pathlib, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"val_SWA": [], "val_CWA": [], "val_SCWA": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- dataset utils -------------------------
def locate_spr() -> pathlib.Path:
    guesses = [
        os.getenv("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for g in guesses:
        p = pathlib.Path(g)
        if (p / "train.csv").exists():
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH not found.")


root = locate_spr()


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _l("train.csv"), "dev": _l("dev.csv"), "test": _l("test.csv")}
    )


spr = load_spr(root)
print({k: len(v) for k, v in spr.items()})


# ---------- tokenisation -----------------
def tokenize(seq: str):
    return seq.strip().split()


vocab = ["<PAD>", "<UNK>"] + sorted(
    {tok for s in spr["train"]["sequence"] for tok in tokenize(s)}
)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = stoi["<PAD>"], stoi["<UNK>"]
labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(labels)}


def encode(seq):
    return [stoi.get(t, unk_idx) for t in tokenize(seq)]


# ---------- metrics ----------------------
def count_shape(seq):
    return len({tok[0] for tok in tokenize(seq)})


def count_color(seq):
    return len({tok[1] for tok in tokenize(seq) if len(tok) > 1})


def SWA(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_t, y_p)) / sum(w)


def CWA(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_t, y_p)) / sum(w)


def SCWA(seqs, y_t, y_p):
    w = [count_shape(s) + count_color(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_t, y_p)) / sum(w)


# ---------- datasets ---------------------
class SPRContrastive(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences

    def __len__(self):
        return len(self.seqs)

    def augment(self, toks):
        toks = toks.copy()
        # random masking
        for i in range(len(toks)):
            if random.random() < 0.15:
                toks[i] = "<UNK>"
        # local swap
        if len(toks) > 1 and random.random() < 0.5:
            i = random.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        return toks

    def __getitem__(self, idx):
        tok = tokenize(self.seqs[idx])
        view1 = self.augment(tok)
        view2 = self.augment(tok)
        return {
            "v1": torch.tensor([stoi.get(t, unk_idx) for t in view1], dtype=torch.long),
            "v2": torch.tensor([stoi.get(t, unk_idx) for t in view2], dtype=torch.long),
        }


def collate_contrastive(batch):
    m1 = max(len(b["v1"]) for b in batch)
    m2 = max(len(b["v2"]) for b in batch)
    v1 = torch.full((len(batch), m1), pad_idx, dtype=torch.long)
    v2 = torch.full((len(batch), m2), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        v1[i, : len(b["v1"])] = b["v1"]
        v2[i, : len(b["v2"])] = b["v2"]
    return {"v1": v1.to(device), "v2": v2.to(device)}


class SPRClassify(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lab = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.seq[idx]), dtype=torch.long)
        return {"ids": ids, "label": torch.tensor(self.lab[idx])}


def collate_classify(batch):
    m = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
    labels = torch.stack([b["label"] for b in batch])
    return {"ids": ids.to(device), "label": labels.to(device)}


# ---------- model ------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.pad = pad_idx

    def forward(self, x):
        mask = (x != self.pad).unsqueeze(-1)
        mean = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return mean


class SimCLR(nn.Module):
    def __init__(self, vocab_size, d_model, proj_dim):
        super().__init__()
        self.enc = Encoder(vocab_size, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        h = self.enc(x)
        z = nn.functional.normalize(self.proj(h), dim=-1)
        return z, h  # return both projection and hidden


# ---------- contrastive loss -------------
def simclr_loss(z1, z2, temp=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2N x D
    sim = torch.matmul(z, z.T) / temp  # cosine since normalized
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])  # positive indices
    loss = nn.CrossEntropyLoss()(sim, labels)
    return loss


# ------------------- training config ---------------------------
batch_c = 256 if torch.cuda.is_available() else 128
batch_f = 256 if torch.cuda.is_available() else 128
d_model = 128
proj_dim = 64
pre_epochs = 5
cls_epochs = 5
print(f"Batch sizes: contrastive={batch_c}, finetune={batch_f}")

# ------------------- pre-training -------------------------------
con_loader = DataLoader(
    SPRContrastive(spr["train"]["sequence"]),
    batch_size=batch_c,
    shuffle=True,
    collate_fn=collate_contrastive,
)
model = SimCLR(len(vocab), d_model, proj_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(1, pre_epochs + 1):
    model.train()
    tot = 0
    for batch in con_loader:
        optimizer.zero_grad()
        z1, _ = model(batch["v1"])
        z2, _ = model(batch["v2"])
        loss = simclr_loss(z1, z2)
        loss.backward()
        optimizer.step()
        tot += loss.item() * batch["v1"].size(0)
    avg = tot / len(con_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(avg)
    print(f"Pretrain epoch {ep}: loss={avg:.4f}")


# ------------------- build classifier ---------------------------
class Classifier(nn.Module):
    def __init__(self, enc, num_cls):
        super().__init__()
        self.enc = enc
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(d_model, num_cls)

    def forward(self, x):
        h = self.enc(x)
        return self.fc(self.drop(h))


clf = Classifier(model.enc, len(labels)).to(device)
opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    SPRClassify(spr["train"]),
    batch_size=batch_f,
    shuffle=True,
    collate_fn=collate_classify,
)
val_loader = DataLoader(
    SPRClassify(spr["dev"]),
    batch_size=batch_f,
    shuffle=False,
    collate_fn=collate_classify,
)

# ------------------- fine-tuning -------------------------------
for ep in range(1, cls_epochs + 1):
    # train
    clf.train()
    tot = 0
    for b in train_loader:
        opt.zero_grad()
        logits = clf(b["ids"])
        loss = criterion(logits, b["label"])
        loss.backward()
        opt.step()
        tot += loss.item() * b["label"].size(0)
    tr_loss = tot / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    # val
    clf.eval()
    v_tot = 0
    preds = []
    trues = []
    with torch.no_grad():
        for b in val_loader:
            logits = clf(b["ids"])
            loss = criterion(logits, b["label"])
            v_tot += loss.item() * b["label"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(b["label"].cpu().tolist())
    v_loss = v_tot / len(val_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    # metrics
    swa = SWA(spr["dev"]["sequence"], trues, preds)
    cwa = CWA(spr["dev"]["sequence"], trues, preds)
    scwa = SCWA(spr["dev"]["sequence"], trues, preds)
    experiment_data["SPR_BENCH"]["metrics"]["val_SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["val_SCWA"].append(scwa)
    print(
        f"Epoch {ep}: val_loss={v_loss:.4f} | SWA {swa:.4f} | CWA {cwa:.4f} | SCWA {scwa:.4f}"
    )

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = trues

# ------------------- save -----------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data.")
