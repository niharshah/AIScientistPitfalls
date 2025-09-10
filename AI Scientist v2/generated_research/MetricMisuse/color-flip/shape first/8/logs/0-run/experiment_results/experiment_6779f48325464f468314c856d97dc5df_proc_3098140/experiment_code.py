import os, random, math, time, pathlib, datetime, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List

# ----------------- setup & device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- attempt to load real SPR_BENCH -----------------
def try_load_real_dataset():
    try:
        from SPR import load_spr_bench  # provided by evaluation infra

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        if not DATA_PATH.exists():
            DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH.")
        return dset
    except Exception as e:
        print("Could not load real SPR_BENCH:", e)
        return None


real_dset = try_load_real_dataset()


# ----------------- synthetic fallback (very small) -----------------
def make_random_token():
    shapes, colors = ["R", "S", "T", "U", "V"], ["A", "B", "C", "D", "E"]
    return random.choice(shapes) + random.choice(colors)


def generate_sequence(min_len=3, max_len=10):
    return " ".join(
        make_random_token() for _ in range(random.randint(min_len, max_len))
    )


def generate_synthetic_split(n):
    return [
        {"id": i, "sequence": generate_sequence(), "label": random.randint(0, 3)}
        for i in range(n)
    ]


if real_dset is None:
    print("Generating synthetic data …")
    real_dset = {
        "train": generate_synthetic_split(2000),
        "dev": generate_synthetic_split(400),
        "test": generate_synthetic_split(400),
    }


# ----------------- metric helpers -----------------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def swa_metric(seqs: List[str], y_t: List[int], y_p: List[int]) -> float:
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def cwa_metric(seqs: List[str], y_t: List[int], y_p: List[int]) -> float:
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def mwa_metric(seqs, y_t, y_p):
    return 0.5 * (swa_metric(seqs, y_t, y_p) + cwa_metric(seqs, y_t, y_p))


# ----------------- vocab & encoding -----------------
PAD, TUNK, TMASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(dataset):
    vocab = set()
    for r in dataset:
        vocab.update(r["sequence"].split())
    vocab_list = [PAD, TUNK, TMASK] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(vocab_list)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size = len(stoi)
MAX_LEN = 20
print("vocab size:", vocab_size)


def encode(seq, max_len=MAX_LEN):
    ids = [stoi.get(tok, stoi[TUNK]) for tok in seq.split()][:max_len]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


# ----------------- datasets -----------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN):
        self.rows = rows
        self.max_len = max_len

    def augment(self, tokens: List[int]):
        toks = [t for t in tokens if t != stoi[PAD]] or [stoi[PAD]]
        if random.random() < 0.3 and len(toks) > 1:
            del toks[random.randint(0, len(toks) - 1)]
        if len(toks) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(toks)), 2)
            toks[i], toks[j] = toks[j], toks[i]
        toks = [stoi[TMASK] if random.random() < 0.15 else t for t in toks]
        seq = " ".join(itos[t] for t in toks)
        return encode(seq, self.max_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = encode(row["sequence"], self.max_len)
        v1 = torch.tensor(self.augment(ids), dtype=torch.long)
        v2 = torch.tensor(self.augment(ids), dtype=torch.long)
        return {"view1": v1, "view2": v2}


class SPRSupervisedDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "ids": torch.tensor(
                encode(row["sequence"], self.max_len), dtype=torch.long
            ),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "seq": row["sequence"],
        }


# ----------------- model -----------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden * 2, 128)

    def forward(self, x):
        emb = self.emb(x)
        h, _ = self.lstm(emb)
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


# -------- contrastive loss --------
def simclr_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], 0)
    sim = (reps @ reps.T) / temp
    mask = torch.eye(2 * B, dtype=torch.bool, device=reps.device)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(B, 2 * B, device=reps.device)
    targets = torch.cat([targets, torch.arange(0, B, device=reps.device)], 0)
    return nn.functional.cross_entropy(sim, targets)


# ------------- evaluation helper -------------
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, batches = 0, 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["ids"])
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item()
            batches += 1
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    swa = swa_metric(seqs, labels, preds)
    cwa = cwa_metric(seqs, labels, preds)
    return tot_loss / max(1, batches), swa, cwa, (swa + cwa) / 2


# ----------------- hyper-parameter settings -----------------
BATCH = 256  # larger batch for better contrastive negatives
FT_EPOCHS = 6
NUM_CLASSES = len({r["label"] for r in real_dset["train"]})
PRE_OPTIONS = [5, 10]  # try two pre-training epoch counts

# persistent loaders
pre_ds = SPRContrastiveDataset(real_dset["train"])
pre_dl = DataLoader(pre_ds, batch_size=BATCH, shuffle=True, drop_last=True)

train_sup_ds = SPRSupervisedDataset(real_dset["train"])
dev_sup_ds = SPRSupervisedDataset(real_dset["dev"])
train_sup_dl = DataLoader(train_sup_ds, batch_size=BATCH, shuffle=True)
dev_sup_dl = DataLoader(dev_sup_ds, batch_size=BATCH, shuffle=False)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_MWA": [], "val_MWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "pretraining_setting": [],
    }
}

for PRE_EPOCHS in PRE_OPTIONS:
    print(f"\n=== Experiment PRE_EPOCHS={PRE_EPOCHS} ===")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # ---- SimCLR pre-training ----
    encoder = Encoder(vocab_size).to(device)
    contrast_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for ep in range(1, PRE_EPOCHS + 1):
        encoder.train()
        tot, batches = 0, 0
        for batch in pre_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = simclr_loss(encoder(batch["view1"]), encoder(batch["view2"]))
            contrast_opt.zero_grad()
            loss.backward()
            contrast_opt.step()
            tot += loss.item()
            batches += 1
        print(f"  PreEpoch {ep}/{PRE_EPOCHS}: loss={tot/batches:.4f}")

    # ---- Fine-tuning ----
    clf = Classifier(encoder, NUM_CLASSES).to(device)
    ft_opt = torch.optim.Adam(clf.parameters(), lr=1.5e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, FT_EPOCHS + 1):
        clf.train()
        tot_loss, batches = 0, 0
        for batch in train_sup_dl:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = clf(batch["ids"])
            loss = criterion(logits, batch["label"])
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()
            tot_loss += loss.item()
            batches += 1
        train_loss = tot_loss / batches

        val_loss, val_swa, val_cwa, val_mwa = evaluate(clf, dev_sup_dl, criterion)
        print(
            f"  FT Epoch {epoch}/{FT_EPOCHS}: val_loss={val_loss:.4f}  MWA={val_mwa:.4f}"
        )

        # ---- logging ----
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train_MWA"].append(
            None
        )  # not computed
        experiment_data["SPR_BENCH"]["metrics"]["val_MWA"].append(val_mwa)
        experiment_data["SPR_BENCH"]["epochs"].append(epoch)
        experiment_data["SPR_BENCH"]["pretraining_setting"].append(PRE_EPOCHS)

# ----------------- save results -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to working/experiment_data.npy")
