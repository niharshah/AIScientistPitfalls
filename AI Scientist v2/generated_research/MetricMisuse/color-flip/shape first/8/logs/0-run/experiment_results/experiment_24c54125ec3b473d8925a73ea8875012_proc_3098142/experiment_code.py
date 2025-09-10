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
        toks = tokens.copy()
        toks = [t for t in toks if t != stoi[PAD]]
        if len(toks) == 0:
            toks = [stoi[PAD]]
        if random.random() < 0.3:
            del_idx = random.randint(0, len(toks) - 1)
            del toks[del_idx]
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
        z = torch.tanh(self.proj(h))
        return z


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


# ----------------- contrastive loss -----------------
def simclr_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(representations, representations.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
    sim_matrix.masked_fill_(mask, -9e15)
    positives = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(
        z1.device
    )
    loss = nn.functional.cross_entropy(sim_matrix, positives)
    return loss


# ----------------- training params -----------------
BATCH = 128
PRE_EPOCHS = 3
FT_EPOCHS = 5
NUM_CLASSES = len(set(r["label"] for r in real_dset["train"]))

# ----------------- dataloaders (constant) -----------------
pretrain_ds_full = SPRContrastiveDataset(real_dset["train"])
train_ds_sup_full = SPRSupervisedDataset(real_dset["train"])
dev_ds_sup_full = SPRSupervisedDataset(real_dset["dev"])

pretrain_dl_full = DataLoader(
    pretrain_ds_full, batch_size=BATCH, shuffle=True, drop_last=True
)
train_dl_sup_full = DataLoader(train_ds_sup_full, batch_size=BATCH, shuffle=True)
dev_dl_sup_full = DataLoader(dev_ds_sup_full, batch_size=BATCH, shuffle=False)

# ----------------- experiment data dict -----------------
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "params": [],  # list of (pre_lr, ft_lr)
            "metrics": {"train_SCWA": [], "val_SCWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ----------------- evaluation helper -----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    loss_sum, batches = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss_sum += loss.item()
            batches += 1
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_seqs.extend(batch["seq"])
    scwa = scwa_metric(all_seqs, all_labels, all_preds)
    return loss_sum / batches, scwa, all_preds, all_labels


# ----------------- learning-rate grid -----------------
pre_lrs = [5e-4, 1e-3, 3e-3, 5e-3]
ft_lrs = [5e-4, 1e-3, 2e-3]

# ----------------- sweep -----------------
for pre_lr, ft_lr in itertools.product(pre_lrs, ft_lrs):
    combo_str = f"pre{pre_lr}_ft{ft_lr}"
    print(f"\n=== Combination {combo_str} ===")
    # reproducibility per run
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # fresh model
    encoder = Encoder(vocab_size).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=pre_lr)

    # ---------- contrastive pre-training ----------
    for epoch in range(1, PRE_EPOCHS + 1):
        encoder.train()
        tot_loss, batches = 0.0, 0
        for batch in pretrain_dl_full:
            v1 = batch["view1"].to(device)
            v2 = batch["view2"].to(device)
            loss = simclr_loss(encoder(v1), encoder(v2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            batches += 1
        print(f"PreEpoch {epoch}: contrastive_loss = {tot_loss/batches:.4f}")

    # ---------- supervised fine-tuning ----------
    clf = Classifier(encoder, NUM_CLASSES).to(device)
    ft_opt = torch.optim.Adam(clf.parameters(), lr=ft_lr)

    for epoch in range(1, FT_EPOCHS + 1):
        clf.train()
        train_loss_sum, batches = 0.0, 0
        for batch in train_dl_sup_full:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device)
            loss = criterion(clf(ids), labels)
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()
            train_loss_sum += loss.item()
            batches += 1
        train_loss = train_loss_sum / batches
        val_loss, val_scwa, preds, gts = evaluate(clf, dev_dl_sup_full)
        print(f"  Epoch {epoch}: val_loss={val_loss:.4f} | SCWA={val_scwa:.4f}")

    # final train-set SCWA for logging
    tr_loss, tr_scwa, _, _ = evaluate(clf, train_dl_sup_full)

    # ---------- log ----------
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
    ed["params"].append((pre_lr, ft_lr))
    ed["metrics"]["train_SCWA"].append(tr_scwa)
    ed["metrics"]["val_SCWA"].append(val_scwa)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["predictions"].append(preds)
    ed["ground_truth"].append(gts)
    ed["timestamps"].append(datetime.datetime.now().isoformat())

# ----------------- save experiment data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
