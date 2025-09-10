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


def scwa_metric(sequences: List[str], y_true: List[int], y_pred: List[int]) -> float:
    weights = [count_shape_variety(s) * count_color_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ----------------- vocab & encoding -----------------
PAD, TUNK, TMASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(dataset):
    vocab = set()
    for row in dataset:
        vocab.update(row["sequence"].split())
    vocab_list = [PAD, TUNK, TMASK] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(vocab_list)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size = len(stoi)
print("vocab size:", vocab_size)


def encode(seq: str, max_len: int):
    ids = [stoi.get(tok, stoi[TUNK]) for tok in seq.split()][:max_len]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


MAX_LEN = 20


# ----------------- datasets -----------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN, supervised=False):
        self.rows = rows
        self.max_len = max_len
        self.supervised = supervised

    def augment(self, tokens: List[int]):
        toks = [t for t in tokens if t != stoi[PAD]]
        if len(toks) == 0:
            toks = [stoi[PAD]]
        if random.random() < 0.3:
            toks.pop(random.randrange(len(toks)))
        if len(toks) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(toks)), 2)
            toks[i], toks[j] = toks[j], toks[i]
        toks = [stoi[TMASK] if random.random() < 0.15 else t for t in toks]
        return encode(" ".join(itos[t] for t in toks), self.max_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = encode(row["sequence"], self.max_len)
        view1 = torch.tensor(self.augment(ids), dtype=torch.long)
        view2 = torch.tensor(self.augment(ids), dtype=torch.long)
        if self.supervised:
            label = torch.tensor(row["label"], dtype=torch.long)
            return {
                "view1": view1,
                "view2": view2,
                "label": label,
                "seq": row["sequence"],
            }
        return {"view1": view1, "view2": view2, "seq": row["sequence"]}


class SPRSupervisedDataset(Dataset):
    def __init__(self, rows, max_len=MAX_LEN):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = torch.tensor(encode(row["sequence"], self.max_len), dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"ids": ids, "label": label, "seq": row["sequence"]}


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
        h = h.transpose(1, 2)
        h = self.pool(h).squeeze(-1)
        return torch.tanh(self.proj(h))


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


# ----------------- contrastive loss -----------------
def simclr_loss(z1, z2, temperature):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    sim = (reps @ reps.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
    sim.masked_fill_(mask, -9e15)
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z1.device)
    return nn.functional.cross_entropy(sim, pos)


# ----------------- fixed data loaders -----------------
BATCH = 128
train_contrastive_ds = SPRContrastiveDataset(real_dset["train"])
train_contrastive_dl = DataLoader(
    train_contrastive_ds, batch_size=BATCH, shuffle=True, drop_last=True
)
train_sup_ds = SPRSupervisedDataset(real_dset["train"])
dev_sup_ds = SPRSupervisedDataset(real_dset["dev"])
train_sup_dl = DataLoader(train_sup_ds, batch_size=BATCH, shuffle=True)
dev_sup_dl = DataLoader(dev_sup_ds, batch_size=BATCH, shuffle=False)

NUM_CLASSES = len(set(r["label"] for r in real_dset["train"]))
PRE_EPOCHS = 3
FT_EPOCHS = 5

# ----------------- experiment data dict -----------------
experiment_data = {"temperature_tuning": {"SPR_BENCH": {}}}

# ----------------- evaluation helper -----------------
criterion_sup = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss, tot_batches = 0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion_sup(logits, labels)
            tot_loss += loss.item()
            tot_batches += 1
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_seqs.extend(batch["seq"])
    scwa = scwa_metric(all_seqs, all_labels, all_preds)
    return tot_loss / tot_batches, scwa, all_preds, all_labels, all_seqs


# ----------------- temperature sweep -----------------
temperatures = [0.1, 0.2, 0.5, 0.7, 1.0]
best_scwa = -1
best_temp = None

for temp in temperatures:
    print(f"\n=== Temperature {temp} ===")
    enc = Encoder(vocab_size).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    # pretrain
    for ep in range(1, PRE_EPOCHS + 1):
        enc.train()
        tot_loss, tot_batches = 0, 0
        for batch in train_contrastive_dl:
            v1 = batch["view1"].to(device)
            v2 = batch["view2"].to(device)
            z1, z2 = enc(v1), enc(v2)
            loss = simclr_loss(z1, z2, temp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            tot_batches += 1
        print(f"PreEpoch {ep}: loss={tot_loss/tot_batches:.4f}")

    # fine-tune
    clf = Classifier(enc, NUM_CLASSES).to(device)
    ft_opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    hist_train_loss, hist_val_loss, hist_val_scwa = [], [], []
    for ep in range(1, FT_EPOCHS + 1):
        clf.train()
        tl, bt = 0, 0
        for batch in train_sup_dl:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = clf(ids)
            loss = criterion_sup(logits, labels)
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()
            tl += loss.item()
            bt += 1
        train_loss = tl / bt
        val_loss, val_scwa, preds, labels_true, seqs = evaluate(clf, dev_sup_dl)
        print(f" FT Epoch {ep}: val_loss={val_loss:.4f} | SCWA={val_scwa:.4f}")
        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_val_scwa.append(val_scwa)
    # store results
    experiment_data["temperature_tuning"]["SPR_BENCH"][temp] = {
        "losses": {"train": hist_train_loss, "val": hist_val_loss},
        "metrics": {"val_SCWA": hist_val_scwa},
        "predictions": preds,
        "ground_truth": labels_true,
        "seqs": seqs,
    }
    if hist_val_scwa[-1] > best_scwa:
        best_scwa = hist_val_scwa[-1]
        best_temp = temp

print(f"\nBest temperature={best_temp} with SCWA={best_scwa:.4f}")

# ----------------- save experiment data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
