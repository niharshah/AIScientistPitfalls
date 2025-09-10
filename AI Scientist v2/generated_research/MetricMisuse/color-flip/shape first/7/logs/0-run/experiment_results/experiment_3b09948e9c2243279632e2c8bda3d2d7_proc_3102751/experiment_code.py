import os, random, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# create working directory and pick device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# deterministic behaviour
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ------------------------------------------------------------------
# locate SPR_BENCH
def find_spr_bench():
    for p in [
        os.environ.get("SPR_DATA_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if p and pathlib.Path(p, "train.csv").exists():
            return pathlib.Path(p).resolve()
    raise FileNotFoundError("SPR_BENCH not found")


DATA_PATH = find_spr_bench()
print("SPR_BENCH:", DATA_PATH)


# ------------------------------------------------------------------
# metrics helpers
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ------------------------------------------------------------------
# load datasets
def load_spr(root):
    load_csv = lambda f: load_dataset(
        "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(
        train=load_csv("train.csv"),
        dev=load_csv("dev.csv"),
        test=load_csv("test.csv"),
    )


spr = load_spr(DATA_PATH)

# ------------------------------------------------------------------
# vocabulary
all_tokens = set(tok for seq in spr["train"]["sequence"] for tok in seq.split())
token2id = {t: i + 2 for i, t in enumerate(sorted(all_tokens))}
PAD, MASK = 0, 1
vocab_size = len(token2id) + 2
num_classes = len(set(spr["train"]["label"]))


def encode(seq):
    return [token2id[t] for t in seq.split()]


# ------------------------------------------------------------------
# datasets
class SPRContrastive(Dataset):
    def __init__(self, sequences, max_len=128):
        self.enc = [encode(s)[:max_len] for s in sequences]

    def __len__(self):
        return len(self.enc)

    def _augment(self, ids):
        ids = ids.copy()
        # 15% masking
        for i in range(len(ids)):
            if random.random() < 0.15:
                ids[i] = MASK
        # local shuffle
        if len(ids) > 4:
            i = random.randint(0, len(ids) - 3)
            j = min(len(ids), i + 3)
            random.shuffle(ids[i:j])
        return ids

    def __getitem__(self, idx):
        ids = self.enc[idx]
        return (
            torch.tensor(self._augment(ids), dtype=torch.long),
            torch.tensor(self._augment(ids), dtype=torch.long),
        )


class SPRClassify(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]
        self.enc = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.enc[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


def pad_collate(batch):
    if isinstance(batch[0], tuple):  # contrastive batch
        a, b = zip(*batch)
        lens = [len(x) for x in a + b]
        mx = max(lens)

        def pad(x):
            return torch.cat([x, torch.full((mx - len(x),), PAD)])

        return torch.stack([pad(x) for x in a]), torch.stack([pad(x) for x in b])
    # classification batch
    ids = [b["input_ids"] for b in batch]
    mx = max(len(x) for x in ids)

    def pad(x):
        return torch.cat([x, torch.full((mx - len(x),), PAD)])

    return {
        "input_ids": torch.stack([pad(x) for x in ids]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw": [b["raw"] for b in batch],
    }


# ------------------------------------------------------------------
# models
class Encoder(nn.Module):
    def __init__(self, emb=64, hid=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)

    def forward(self, x):
        emb = self.embed(x)
        mask = x != PAD
        lens = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[-2], h[-1]], 1)  # (B, 2*hid)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, proj),
            nn.ReLU(),
            nn.Linear(proj, proj),
        )

    def forward(self, x):
        return self.fc(x)


class Classifier(nn.Module):
    def __init__(self, encoder, hidden, classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ------------------------------------------------------------------
# fixed SimCLR / InfoNCE loss (vectorised)
def simclr_loss(z1, z2, temperature=0.07):
    z = torch.cat([z1, z2], dim=0)  # (2N, d)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature  # (2N,2N)
    N = z1.size(0)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)  # remove self-similarities
    pos_indices = torch.arange(N, 2 * N, device=z.device)
    target = torch.cat([pos_indices, torch.arange(0, N, device=z.device)], dim=0)
    loss = nn.functional.cross_entropy(sim, target)
    return loss


# ------------------------------------------------------------------
# experiment container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------
# contrastive pre-training
BATCH = 256
contr_loader = DataLoader(
    SPRContrastive(spr["train"]["sequence"]),
    batch_size=BATCH,
    shuffle=True,
    collate_fn=pad_collate,
)
encoder = Encoder().to(device)
proj = ProjectionHead(256).to(device)
optim_c = torch.optim.Adam(
    list(encoder.parameters()) + list(proj.parameters()), lr=1e-3
)

print("Contrastive pre-training...")
for epoch in range(2):  # short demo run
    tot, nb = 0.0, 0
    encoder.train()
    proj.train()
    for v1, v2 in contr_loader:
        v1, v2 = v1.to(device), v2.to(device)
        optim_c.zero_grad()
        z1, z2 = proj(encoder(v1)), proj(encoder(v2))
        loss = simclr_loss(z1, z2)
        loss.backward()
        optim_c.step()
        tot += loss.item()
        nb += 1
    print(f"Pre-train epoch {epoch+1}: loss = {tot/nb:.4f}")

# ------------------------------------------------------------------
# fine-tuning
train_loader = DataLoader(
    SPRClassify(spr["train"]), batch_size=128, shuffle=True, collate_fn=pad_collate
)
dev_loader = DataLoader(
    SPRClassify(spr["dev"]), batch_size=256, shuffle=False, collate_fn=pad_collate
)

clf = Classifier(encoder, 128, num_classes).to(device)
optim_f = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    # ------------- train -------------
    clf.train()
    tot, nb = 0.0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optim_f.zero_grad()
        logit = clf(batch["input_ids"])
        loss = criterion(logit, batch["label"])
        loss.backward()
        optim_f.step()
        tot += loss.item()
        nb += 1
    tr_loss = tot / nb
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))

    # ------------- validation -------------
    clf.eval()
    tot, nb = 0.0, 0
    preds, gts, seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logit = clf(batch["input_ids"])
            loss = criterion(logit, batch["label"])
            tot += loss.item()
            nb += 1
            p = logit.argmax(1).cpu().tolist()
            preds.extend(p)
            g = batch["label"].cpu().tolist()
            gts.extend(g)
            seqs.extend(batch["raw"])
    val_loss = tot / nb
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))

    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    dawa = (swa + cwa) / 2
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, swa, cwa, dawa))

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}  SWA={swa:.4f} "
        f"CWA={cwa:.4f} DAWA={dawa:.4f}"
    )

# ------------------------------------------------------------------
# save predictions/ground truth of last epoch
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
