import os, random, math, time, pathlib, itertools, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- try loading SPR_BENCH -----------------
def try_load_real_dataset():
    try:
        from SPR import load_spr_bench  # local util provided by task

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        if not DATA_PATH.exists():
            DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH.")
        return dset
    except Exception as e:
        print("Could not load real SPR_BENCH: ", e)
        return None


real_dset = try_load_real_dataset()


# ----------------- synthetic fallback -----------------
def make_random_token():
    shapes = ["R", "S", "T", "U", "V"]
    colors = ["A", "B", "C", "D", "E"]
    return random.choice(shapes) + random.choice(colors)


def generate_sequence(min_len=3, max_len=10):
    return " ".join(
        make_random_token() for _ in range(random.randint(min_len, max_len))
    )


def generate_synthetic_split(n_rows: int):
    data = []
    for i in range(n_rows):
        seq = generate_sequence()
        label = random.randint(0, 3)
        data.append({"id": i, "sequence": seq, "label": label})
    return data


if real_dset is None:
    print("Generating synthetic data …")
    real_dset = {
        "train": generate_synthetic_split(1000),
        "dev": generate_synthetic_split(200),
        "test": generate_synthetic_split(200),
    }


# ----------------- SCWA metric helpers -----------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split()))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split()))


def scwa_metric(seqs: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ----------------- vocab & encoding -----------------
PAD, TUNK, TMASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(dataset):
    vocab = set()
    for r in dataset:
        vocab.update(r["sequence"].split())
    vocab_list = [PAD, TUNK, TMASK] + sorted(vocab)
    stoi = {t: i for i, t in enumerate(vocab_list)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size = len(stoi)
print("vocab size:", vocab_size)


def encode(seq: str, max_len: int):
    ids = [stoi.get(tok, stoi[TUNK]) for tok in seq.split()][:max_len]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


MAX_LEN = 20


# ----------------- datasets -----------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN, supervised=False):
        self.rows, self.max_len, self.supervised = rows, max_len, supervised

    def augment(self, tokens: List[int]):
        toks = [t for t in tokens if t != stoi[PAD]] or [stoi[PAD]]
        # deletion
        if len(toks) > 1 and random.random() < 0.3:
            del toks[random.randrange(len(toks))]
        # swap
        if len(toks) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(toks)), 2)
            toks[i], toks[j] = toks[j], toks[i]
        # mask
        toks = [stoi[TMASK] if random.random() < 0.15 else t for t in toks]
        return encode(" ".join(itos[t] for t in toks), self.max_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = encode(row["sequence"], self.max_len)
        v1, v2 = torch.tensor(self.augment(ids)), torch.tensor(self.augment(ids))
        if self.supervised:
            return {
                "view1": v1,
                "view2": v2,
                "label": torch.tensor(row["label"]),
                "seq": row["sequence"],
            }
        return {"view1": v1, "view2": v2, "seq": row["sequence"]}


class SPRSupervisedDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "ids": torch.tensor(encode(r["sequence"], self.max_len)),
            "label": torch.tensor(r["label"]),
            "seq": r["sequence"],
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
        h, _ = self.lstm(self.emb(x))
        h = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


# ----------------- contrastive loss -----------------
def simclr_loss(z1, z2, T=0.5):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(reps, reps.T) / T
    sim.fill_diagonal_(-9e15)
    pos = torch.arange(B, 2 * B).tolist() + torch.arange(0, B).tolist()
    return nn.functional.cross_entropy(sim, torch.tensor(pos, device=z1.device))


# ----------------- misc params -----------------
BATCH, PRE_EPOCHS = 128, 3
NUM_CLASSES = len({r["label"] for r in real_dset["train"]})
print("num classes:", NUM_CLASSES)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ----------------- dataloaders -----------------
pretrain_dl = DataLoader(
    SPRContrastiveDataset(real_dset["train"]),
    batch_size=BATCH,
    shuffle=True,
    drop_last=True,
)
train_dl_sup = DataLoader(
    SPRSupervisedDataset(real_dset["train"]), batch_size=BATCH, shuffle=True
)
dev_dl_sup = DataLoader(
    SPRSupervisedDataset(real_dset["dev"]), batch_size=BATCH, shuffle=False
)

# ----------------- experiment data dict -----------------
experiment_data = {
    "FT_EPOCHS": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
            "ft_setting": [],
            "timestamps": [],
        }
    }
}

# ----------------- pretraining (once) -----------------
encoder_base = Encoder(vocab_size).to(device)
opt = torch.optim.Adam(encoder_base.parameters(), lr=1e-3)
for epoch in range(1, PRE_EPOCHS + 1):
    encoder_base.train()
    tot = 0
    n = 0
    for batch in pretrain_dl:
        z1 = encoder_base(batch["view1"].to(device))
        z2 = encoder_base(batch["view2"].to(device))
        loss = simclr_loss(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
    print(f"Pretrain Epoch {epoch}: loss {tot/n:.4f}")
pretrained_state = {k: v.cpu() for k, v in encoder_base.state_dict().items()}
del encoder_base  # free memory

# ----------------- evaluate helper -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss = 0
    n = 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            y = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, y)
            tot_loss += loss.item()
            n += 1
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            seqs.extend(batch["seq"])
    return tot_loss / n, scwa_metric(seqs, labels, preds), preds, labels


# ----------------- hyperparameter tuning for FT_EPOCHS -----------------
FT_CANDIDATES = [5, 10, 15, 20]
patience = 3
for ft_epochs in FT_CANDIDATES:
    print(f"\n=== Fine-tuning for {ft_epochs} epochs ===")
    enc = Encoder(vocab_size).to(device)
    enc.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
    clf = Classifier(enc, NUM_CLASSES).to(device)
    ft_opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    best_scwa, epochs_no_improve = -1, 0
    for epoch in range(1, ft_epochs + 1):
        clf.train()
        tot = 0
        n = 0
        for batch in train_dl_sup:
            ids = batch["ids"].to(device)
            y = batch["label"].to(device)
            loss = criterion(clf(ids), y)
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()
            tot += loss.item()
            n += 1
        train_loss = tot / n
        val_loss, val_scwa, preds, gts = evaluate(clf, dev_dl_sup)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f} SCWA={val_scwa:.4f}")
        # logging
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["metrics"]["train"].append(None)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["metrics"]["val"].append(val_scwa)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["epochs"].append(epoch)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["ft_setting"].append(ft_epochs)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["predictions"].append(preds)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["ground_truth"].append(gts)
        experiment_data["FT_EPOCHS"]["SPR_BENCH"]["timestamps"].append(
            datetime.datetime.now().isoformat()
        )
        # early stopping
        if val_scwa > best_scwa:
            best_scwa = val_scwa
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# ----------------- save experiment data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
