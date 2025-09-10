# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, math, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

# -------------------------------------------------
# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# Device handling (MUST)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------
# ----- SPR helper functions (copied from SPR.py) --
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# -------------------------------------------------
# --------------- Dataset preparation -------------
data_root_candidates = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print(f"Loaded data from {p}")
        break
if spr_bench is None:
    # fallback small synthetic dataset to keep script runnable
    print("SPR_BENCH not found, generating tiny synthetic data.")

    def synth(n):
        seqs, labels = [], []
        shapes = "ABCD"
        colors = "1234"
        for _ in range(n):
            L = random.randint(4, 9)
            seq = []
            for _ in range(L):
                seq.append(random.choice(shapes) + random.choice(colors))
            seqs.append(" ".join(seq))
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for split, n in [("train", 500), ("dev", 100), ("test", 100)]:
        spr_bench[split] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )


# Build vocabulary
def tokenize(seq):
    return seq.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size = len(vocab)
mask_id = vocab["<MASK>"]
print(f"Vocab size = {vocab_size}")


# -------------------------------------------------
# --------- PyTorch datasets ----------------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def _augment(self, toks, p=0.15):
        out = []
        for t in toks:
            r = random.random()
            if r < p:  # drop
                continue
            out.append(t)
        if len(out) == 0:
            out.append(random.choice(toks))
        return out

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        view1 = self._augment(toks)
        view2 = self._augment(toks)
        return view1, view2


class SPRClassifierDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


# Collate fns
def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def encode(list_tokens):
        ids = [
            torch.tensor([vocab[t] for t in toks], dtype=torch.long)
            for toks in list_tokens
        ]
        lens = [len(i) for i in ids]
        ids = pad_sequence(ids, batch_first=True, padding_value=0)
        return ids, torch.tensor(lens, dtype=torch.long)

    ids1, len1 = encode(v1)
    ids2, len2 = encode(v2)
    return {
        "ids1": ids1.to(device),
        "len1": len1.to(device),
        "ids2": ids2.to(device),
        "len2": len2.to(device),
    }


def collate_classifier(batch):
    toks, labels, raw_seq = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in t], dtype=torch.long) for t in toks]
    lens = [len(i) for i in ids]
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    return {
        "ids": ids.to(device),
        "len": torch.tensor(lens, dtype=torch.long).to(device),
        "label": torch.tensor(labels, dtype=torch.long).to(device),
        "sequence": raw_seq,
    }


# DataLoaders
pretrain_loader = DataLoader(
    SPRContrastiveDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    SPRClassifierDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    SPRClassifierDataset(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)


# -------------------------------------------------
# --------- Model definition ----------------------
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2H)
        z = self.proj(h)
        return z


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj.out_features, num_classes)

    def forward(self, ids, lens):
        z = self.encoder(ids, lens)
        return self.head(z)


# Instantiate
encoder = Encoder(vocab).to(device)
clf_model = Classifier(encoder, num_classes=len(set(spr_bench["train"]["label"]))).to(
    device
)


# -------------------------------------------------
# --------- Contrastive loss ----------------------
def nt_xent_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x D
    sim = torch.matmul(z, z.T) / temp  # 2N x 2N
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    logits_12 = sim[:N, N:]  # positives across views
    logits_21 = sim[N:, :N]
    denom_12 = torch.logsumexp(sim[:N], dim=1)
    denom_21 = torch.logsumexp(sim[N:], dim=1)
    loss = (-logits_12.diag() + denom_12 + -logits_21.diag() + denom_21).mean() * 0.5
    return loss


# -------------------------------------------------
# --------- Training utilities --------------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCHM": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --- Pretraining ---------------------------------
optimizer_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
pretrain_epochs = 2
for epoch in range(1, pretrain_epochs + 1):
    encoder.train()
    epoch_loss = 0.0
    for batch in pretrain_loader:
        optimizer_pt.zero_grad()
        z1 = encoder(batch["ids1"], batch["len1"])
        z2 = encoder(batch["ids2"], batch["len2"])
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        optimizer_pt.step()
        epoch_loss += loss.item() * batch["ids1"].size(0)
    epoch_loss /= len(pretrain_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(epoch_loss)
    print(f"Pretrain Epoch {epoch}: loss={epoch_loss:.4f}")

# --- Fine-tuning ---------------------------------
optimizer_ft = torch.optim.Adam(clf_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
ft_epochs = 3
for epoch in range(1, ft_epochs + 1):
    # train
    clf_model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer_ft.zero_grad()
        logits = clf_model(batch["ids"], batch["len"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item() * batch["ids"].size(0)
    train_loss = running_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    # eval
    clf_model.eval()
    val_loss = 0.0
    all_pred, all_true, all_seq = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            logits = clf_model(batch["ids"], batch["len"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["ids"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(batch["label"].cpu().tolist())
            all_seq.extend(batch["sequence"])
    val_loss /= len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    schm = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["SCHM"].append(schm)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCHM={schm:.3f}"
    )

# -------------------------------------------------
# ---------- Save experiment data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ---------- Visualization ------------------------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Fine-tune loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
print("Saved experiment data and plot in ./working/")
