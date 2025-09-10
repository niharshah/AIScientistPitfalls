import os, pathlib, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from collections import Counter

# --------------------------------------------------------------
# working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------
# misc helpers
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(0)


# --------------------------------------------------------------
# locate and load SPR_BENCH
def locate_spr():
    candidates = [os.getenv("SPR_BENCH_PATH", "")]
    here = pathlib.Path.cwd()
    candidates += [
        here / "SPR_BENCH",
        here.parent / "SPR_BENCH",
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for c in candidates:
        p = pathlib.Path(c)
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError("SPR_BENCH CSVs not found")


root = locate_spr()


def _load(csv):
    return load_dataset(
        "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
    )


spr = DatasetDict(
    train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
)
print({k: len(v) for k, v in spr.items()})


# --------------------------------------------------------------
# tokenisation & vocab
def tok(seq):
    return seq.strip().split()


all_tokens = [t for s in spr["train"]["sequence"] for t in tok(s)]
vocab = ["<PAD>", "<UNK>"] + sorted(Counter(all_tokens))
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = 0, 1


def enc(seq):
    return [stoi.get(t, unk_idx) for t in tok(seq)]


labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(labels)}


# --------------------------------------------------------------
# datasets
class SPRCls(Dataset):
    def __init__(self, split, with_label=True):
        self.seqs = split["sequence"]
        self.with_label = with_label
        if with_label:
            self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        item = {"ids": torch.tensor(enc(self.seqs[idx]), dtype=torch.long)}
        if self.with_label:
            item["label"] = torch.tensor(self.labels[idx])
        return item


def pad_collate(batch):
    maxlen = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
    out = {"input_ids": ids}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


train_loader_cls = DataLoader(
    SPRCls(spr["train"]), batch_size=128, shuffle=True, collate_fn=pad_collate
)
val_loader_cls = DataLoader(
    SPRCls(spr["dev"]), batch_size=256, shuffle=False, collate_fn=pad_collate
)

# contrastive dataloader (no labels, bigger batch)
pretrain_loader = DataLoader(
    SPRCls(spr["train"], with_label=False),
    batch_size=256,
    shuffle=True,
    collate_fn=pad_collate,
)


# --------------------------------------------------------------
# augmentation
def augment(x):
    # x: [seq_len] tensor
    ids = x.clone()
    # mask 15%
    mask_prob = 0.15
    mask = torch.rand_like(ids.float()) < mask_prob
    ids[mask] = unk_idx
    # local shuffle 10%
    if random.random() < 0.1 and len(ids) > 3:
        i = random.randint(1, len(ids) - 2)
        ids[i], ids[i + 1] = ids[i + 1].clone(), ids[i].clone()
    return ids


# --------------------------------------------------------------
# models
class Encoder(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        mask = (x != pad_idx).unsqueeze(-1)
        e = self.emb(x)
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.drop(mean)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, proj_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, z):
        return self.mlp(z)


class Classifier(nn.Module):
    def __init__(self, encoder, feat_dim, n_labels):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(feat_dim, n_labels)

    def forward(self, x):
        return self.fc(self.enc(x))


# --------------------------------------------------------------
# contrastive loss (SimCLR)
def nt_xent(z1, z2, temp=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    pos = torch.cat(
        [torch.arange(N, device=z.device) + N, torch.arange(N, device=z.device)], dim=0
    )
    loss = F.cross_entropy(sim, pos)
    return loss


# --------------------------------------------------------------
# pre-training
feat_dim = 128
encoder = Encoder(len(vocab), feat_dim).to(device)
proj = ProjectionHead(feat_dim, feat_dim).to(device)
opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=3e-3)

print("--- Contrastive pre-training ---")
for epoch in range(1, 3):  # 2 epochs
    encoder.train()
    proj.train()
    epoch_loss = 0
    for batch in pretrain_loader:
        ids = batch["input_ids"].to(device)
        v1 = torch.stack([augment(seq) for seq in ids]).to(device)
        v2 = torch.stack([augment(seq) for seq in ids]).to(device)
        z1 = proj(encoder(v1))
        z2 = proj(encoder(v2))
        loss = nt_xent(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * ids.size(0)
    print(f"Pretrain epoch {epoch}: loss={epoch_loss/len(pretrain_loader.dataset):.4f}")

# --------------------------------------------------------------
# fine-tuning for classification
clf = Classifier(encoder, feat_dim, len(labels)).to(device)
optim_ft = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()


# metrics helpers
def count_shape(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def SCWA(seqs, true, pred):
    w = [count_shape(s) + count_color(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, true, pred)]
    return sum(corr) / sum(w)


experiment_data = {
    "contrastive_ft": {
        "losses": {"train": [], "val": []},
        "metrics": {"SCWA_train": [], "SCWA_val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

print("--- Supervised fine-tuning ---")
for epoch in range(1, 4):  # 3 epochs
    # train
    clf.train()
    train_loss, train_pred, train_true = 0, [], []
    for batch in train_loader_cls:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = clf(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        optim_ft.zero_grad()
        loss.backward()
        optim_ft.step()
        train_loss += loss.item() * batch["label"].size(0)
        train_pred += logits.argmax(1).cpu().tolist()
        train_true += batch["label"].cpu().tolist()
    train_loss /= len(train_loader_cls.dataset)
    train_scwa = SCWA(spr["train"]["sequence"], train_true, train_pred)

    # validation
    clf.eval()
    val_loss, val_pred, val_true = 0, [], []
    with torch.no_grad():
        for batch in val_loader_cls:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = clf(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["label"].size(0)
            val_pred += logits.argmax(1).cpu().tolist()
            val_true += batch["label"].cpu().tolist()
    val_loss /= len(val_loader_cls.dataset)
    val_scwa = SCWA(spr["dev"]["sequence"], val_true, val_pred)

    experiment_data["contrastive_ft"]["losses"]["train"].append(train_loss)
    experiment_data["contrastive_ft"]["losses"]["val"].append(val_loss)
    experiment_data["contrastive_ft"]["metrics"]["SCWA_train"].append(train_scwa)
    experiment_data["contrastive_ft"]["metrics"]["SCWA_val"].append(val_scwa)
    experiment_data["contrastive_ft"]["predictions"] = val_pred
    experiment_data["contrastive_ft"]["ground_truth"] = val_true
    experiment_data["contrastive_ft"]["timestamps"].append(time.time())

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SCWA = {val_scwa:.4f}")

# --------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
