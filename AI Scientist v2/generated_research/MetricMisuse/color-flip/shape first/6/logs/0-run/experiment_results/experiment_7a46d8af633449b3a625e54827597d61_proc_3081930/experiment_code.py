import os, random, csv, pathlib, math, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset

# ---------- mandatory work dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device handling ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "AIS": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epoch_timestamps": [],
    }
}

# ---------- dataset loading utility (from prompt) ----------
import pathlib as _pl


def load_spr_bench(root: _pl.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


# ---------- fallback synthetic data ----------
def create_synthetic_spr(root):
    root.mkdir(parents=True, exist_ok=True)
    shapes = list("ABCDE")
    colors = list("rgbcy")

    def rand_seq():
        L = random.randint(5, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    for split, n in [("train", 400), ("dev", 100), ("test", 100)]:
        with open(root / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                seq = rand_seq()
                label = random.randint(0, 1)
                w.writerow([i, seq, label])


# ---------- locate dataset ----------
DATA_PATH = pathlib.Path(os.getcwd()) / "SPR_BENCH"
if not DATA_PATH.exists():
    print("Dataset folder not found – creating small synthetic benchmark.")
    create_synthetic_spr(DATA_PATH)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- simple tokenizer ----------
PAD, CLS = "[PAD]", "[CLS]"


def build_vocab(dataset):
    vocab = set()
    for seq in dataset["train"]["sequence"]:
        vocab.update(seq.split())
    vocab = [PAD, CLS] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return stoi, vocab


stoi, vocab = build_vocab(spr)
vocab_size = len(vocab)


def encode(seq):
    return [stoi[CLS]] + [stoi[tok] for tok in seq.split()]


# ---------- augmentation ----------
def augment(seq_tokens):
    # token drop 0.15 + adjacent swap 0.15
    toks = seq_tokens.copy()
    # drop
    toks = [t for t in toks if (t == stoi[CLS] or random.random() > 0.15)]
    if len(toks) < 2:
        toks = seq_tokens
    # swap
    i = 1
    while i < len(toks) - 1:
        if random.random() < 0.15:
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
            i += 2
        else:
            i += 1
    return toks


# ---------- dataset classes ----------
class ContrastiveDS(Dataset):
    def __init__(self, sequences):
        self.seqs = [encode(s) for s in sequences]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        base = self.seqs[idx]
        v1 = augment(base)
        v2 = augment(base)
        return {"view1": torch.tensor(v1), "view2": torch.tensor(v2)}


class SupervisedDS(Dataset):
    def __init__(self, sequences, labels):
        self.data = [encode(s) for s in sequences]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.data[idx]),
            "label": torch.tensor(self.labels[idx]),
        }


def pad_batch(batch, key):
    lens = [len(x[key]) for x in batch]
    maxlen = max(lens)
    for b in batch:
        pad_len = maxlen - len(b[key])
        b[key] = torch.cat([b[key], torch.full((pad_len,), stoi[PAD])])
    return torch.stack([b[key] for b in batch]), torch.tensor(lens)


def collate_contrastive(batch):
    v1, _ = pad_batch(batch, "view1")
    v2, _ = pad_batch(batch, "view2")
    return {"v1": v1, "v2": v2}


def collate_supervised(batch):
    seqs, lens = pad_batch(batch, "seq")
    labels = torch.stack([b["label"] for b in batch]).long()
    return {"seqs": seqs, "lens": lens, "labels": labels}


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=128, layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=stoi[PAD])
        self.lstm = nn.LSTM(
            emb_dim, hid, num_layers=layers, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(hid * 2, 128)

    def forward(self, x, lens):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        z = torch.relu(self.proj(h))
        z = nn.functional.normalize(z, dim=1)
        return z


class Classifier(nn.Module):
    def __init__(self, enc, num_classes=2):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, lens):
        with torch.set_grad_enabled(self.training):
            z = self.enc(x, lens)
        return self.fc(z)


# ---------- contrastive loss ----------
def info_nce(z1, z2, temp=0.07):
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2N x d
    sim = z @ z.t() / temp
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)
    loss = nn.functional.cross_entropy(sim, labels)
    return loss


# ---------- build dataloaders ----------
batch_size = 64
contr_ds = ContrastiveDS(spr["train"]["sequence"])
contr_loader = DataLoader(
    contr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_contrastive
)

sup_train_ds = SupervisedDS(spr["train"]["sequence"], spr["train"]["label"])
sup_dev_ds = SupervisedDS(spr["dev"]["sequence"], spr["dev"]["label"])
train_loader = DataLoader(
    sup_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_supervised
)
dev_loader = DataLoader(
    sup_dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_supervised
)

# ---------- instantiate models ----------
encoder = Encoder(vocab_size).to(device)
proj_params = list(encoder.parameters())
optim_contr = torch.optim.Adam(proj_params, lr=1e-3)

# ---------- contrastive pre-train ----------
epochs_contr = 1
for epoch in range(epochs_contr):
    encoder.train()
    total = 0
    for batch in contr_loader:
        v1 = batch["v1"].to(device)
        v2 = batch["v2"].to(device)
        l1 = (v1 != stoi[PAD]).sum(1)
        l2 = (v2 != stoi[PAD]).sum(1)
        z1 = encoder(v1, l1)
        z2 = encoder(v2, l2)
        loss = info_nce(z1, z2)
        optim_contr.zero_grad()
        loss.backward()
        optim_contr.step()
        total += loss.item()
    print(f"Contrastive epoch {epoch}: loss={total/len(contr_loader):.4f}")

# ---------- supervised fine-tuning ----------
clf = Classifier(encoder).to(device)
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def evaluate():
    clf.eval()
    val_loss = 0
    preds = []
    gts = []
    with torch.no_grad():
        for batch in dev_loader:
            seqs = batch["seqs"].to(device)
            lens = batch["lens"].to(device)
            labels = batch["labels"].to(device)
            logits = clf(seqs, lens)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(labels.cpu().tolist())
    return val_loss / len(dev_loader), preds, gts


def augmentation_views(seq):
    toks = encode(seq)
    return [augment(toks) for _ in range(3)]


def compute_AIS(pred_fn, sequences):
    consistent = 0
    for s in sequences:
        base_pred = pred_fn(
            torch.tensor(encode(s)).unsqueeze(0).to(device),
            torch.tensor([len(encode(s))], device=device),
        )
        views = augmentation_views(s)
        ok = True
        for v in views:
            p = pred_fn(
                torch.tensor(v).unsqueeze(0).to(device),
                torch.tensor([len(v)], device=device),
            )
            if p != base_pred:
                ok = False
                break
        if ok:
            consistent += 1
    return consistent / len(sequences)


def infer_pred(x, lens):
    with torch.no_grad():
        logit = clf(x, lens)
        return logit.argmax(1).item()


epochs_sup = 3
for epoch in range(1, epochs_sup + 1):
    clf.train()
    total_loss = 0
    for batch in train_loader:
        seqs = batch["seqs"].to(device)
        lens = batch["lens"].to(device)
        labels = batch["labels"].to(device)
        logits = clf(seqs, lens)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train = total_loss / len(train_loader)

    val_loss, preds, gts = evaluate()
    # compute AIS on dev subset (first 200)
    sample_seqs = spr["dev"]["sequence"][:200]
    AIS = compute_AIS(infer_pred, sample_seqs)

    print(
        f"Epoch {epoch}: train_loss={avg_train:.4f}  validation_loss = {val_loss:.4f}  AIS={AIS:.3f}"
    )

    # log
    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append(avg_train)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["AIS"].append(AIS)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(avg_train)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    if epoch == epochs_sup:  # final epoch store predictions/gt
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = gts
    experiment_data["SPR_BENCH"]["epoch_timestamps"].append(time.time())

# ---------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
