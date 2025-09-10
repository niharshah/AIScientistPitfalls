import os, random, math, time, datetime, itertools, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

# ──────────────────── set up work dir ────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ──────────────────── device ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────── try loading real SPR_BENCH ─────────
def try_load_real_dataset():
    try:
        from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        if not DATA_PATH.exists():
            DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH.")
        return dset, shape_weighted_accuracy, color_weighted_accuracy
    except Exception as e:
        print("Could not load real SPR_BENCH:", e)
        return None, None, None


real_dset, shape_weighted_accuracy, color_weighted_accuracy = try_load_real_dataset()


# ──────────────────── synthetic fallback ─────────────────
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
        "train": generate_synthetic_split(1000),
        "dev": generate_synthetic_split(200),
        "test": generate_synthetic_split(200),
    }

# ────────────────── metric fallbacks ─────────────────────
if shape_weighted_accuracy is None:

    def _count_shape_variety(seq: str) -> int:
        return len(set(tok[0] for tok in seq.split() if tok))

    def _count_color_variety(seq: str) -> int:
        return len(set(tok[1] for tok in seq.split() if tok))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        weights = [_count_shape_variety(s) for s in seqs]
        correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
        return sum(correct) / (sum(weights) or 1)

    def color_weighted_accuracy(seqs, y_true, y_pred):
        weights = [_count_color_variety(s) for s in seqs]
        correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
        return sum(correct) / (sum(weights) or 1)


def mwa_metric(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return (swa + cwa) / 2.0


# ─────────────────── vocab utils ─────────────────────────
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


# ─────────────────── data sets ───────────────────────────
class SPRContrastiveDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN):
        self.rows = rows
        self.max_len = max_len

    def augment(self, tokens: List[int]):
        toks = [t for t in tokens if t != stoi[PAD]] or [stoi[PAD]]
        # drop
        if len(toks) > 1 and random.random() < 0.3:
            del toks[random.randint(0, len(toks) - 1)]
        # swap
        if len(toks) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(toks)), 2)
            toks[i], toks[j] = toks[j], toks[i]
        # mask
        toks = [stoi[TMASK] if random.random() < 0.15 else t for t in toks]
        s = " ".join(itos[t] for t in toks)
        return encode(s, self.max_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        token_ids = encode(row["sequence"], self.max_len)
        v1 = torch.tensor(self.augment(token_ids), dtype=torch.long)
        v2 = torch.tensor(self.augment(token_ids), dtype=torch.long)
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


# ─────────────────── model ───────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden * 2, 128)

    def forward(self, x):
        x = self.emb(x)
        h, _ = self.lstm(x)
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, num_classes)

    def forward(self, ids):
        return self.head(self.encoder(ids))


# ─────────────────── SimCLR loss ─────────────────────────
def simclr_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = (reps @ reps.T) / temp  # 2B x 2B
    mask = torch.eye(2 * B, dtype=torch.bool, device=reps.device)
    sim.masked_fill_(mask, -9e15)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(reps.device)
    return nn.functional.cross_entropy(sim, targets)


# ───────────────── evaluation helper ─────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, batches = 0.0, 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            # move to device
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["ids"])
            loss = criterion(logits, batch_t["label"])
            tot_loss += loss.item()
            batches += 1
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(batch_t["label"].cpu().tolist())
            seqs.extend(batch_t["seq"])
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    mwa = (swa + cwa) / 2.0
    return tot_loss / batches, swa, cwa, mwa, preds, labels, seqs


# ────────────────── training config ─────────────────────
BATCH = 128
FT_EPOCHS = 10
NUM_CLASSES = len({r["label"] for r in real_dset["train"]})
PRE_OPTIONS = [3, 8, 15]  # hyper-parameter grid

# common dataloaders
pre_dl = DataLoader(
    SPRContrastiveDataset(real_dset["train"]),
    batch_size=BATCH,
    shuffle=True,
    drop_last=True,
)
train_sup_dl = DataLoader(
    SPRSupervisedDataset(real_dset["train"]), batch_size=BATCH, shuffle=True
)
dev_sup_dl = DataLoader(
    SPRSupervisedDataset(real_dset["dev"]), batch_size=BATCH, shuffle=False
)

experiment_data: Dict[str, Dict] = {}

# ─────────────────── experiments ─────────────────────────
for PRE_EPOCHS in PRE_OPTIONS:
    tag = f"PRE={PRE_EPOCHS}"
    print(f"\n=== Experiment {tag} ===")
    # reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # build encoder & optimizer
    encoder = Encoder(vocab_size).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    # ――― pre-training ―――
    for epoch in range(1, PRE_EPOCHS + 1):
        encoder.train()
        tot, batches = 0.0, 0
        for batch in pre_dl:
            v1, v2 = batch["view1"].to(device), batch["view2"].to(device)
            loss = simclr_loss(encoder(v1), encoder(v2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            batches += 1
        print(f"  PreEpoch {epoch}/{PRE_EPOCHS} loss={tot/batches:.4f}")
    # ――― fine-tuning ―――
    clf = Classifier(encoder, NUM_CLASSES).to(device)
    ft_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # prepare logging dict
    experiment_data[tag] = {
        "metrics": {"train_MWA": [], "val_MWA": [], "val_SWA": [], "val_CWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, FT_EPOCHS + 1):
        clf.train()
        tot, batches = 0.0, 0
        for batch in train_sup_dl:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = clf(batch_t["ids"])
            loss = criterion(logits, batch_t["label"])
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()
            tot += loss.item()
            batches += 1
        train_loss = tot / batches
        # validation
        val_loss, swa, cwa, mwa, preds, labels_true, seqs = evaluate(
            clf, dev_sup_dl, criterion
        )
        # logging
        ts = datetime.datetime.now().isoformat()
        experiment_data[tag]["losses"]["train"].append(train_loss)
        experiment_data[tag]["losses"]["val"].append(val_loss)
        experiment_data[tag]["metrics"]["train_MWA"].append(None)
        experiment_data[tag]["metrics"]["val_MWA"].append(mwa)
        experiment_data[tag]["metrics"]["val_SWA"].append(swa)
        experiment_data[tag]["metrics"]["val_CWA"].append(cwa)
        experiment_data[tag]["predictions"] = preds
        experiment_data[tag]["ground_truth"] = labels_true
        experiment_data[tag]["timestamps"].append(ts)
        print(
            f"  FT Epoch {epoch}/{FT_EPOCHS}: val_loss={val_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} MWA={mwa:.4f}"
        )
# ─────────────────── save results ────────────────────────
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
