# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, string, pathlib, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# ----------------------------------------- experiment bookkeeping ---------
experiment_data = {
    "last_token_repr": {
        "SPR": {
            "contrastive_pretrain": {"losses": []},
            "fine_tune": {
                "losses": {"train": [], "val": []},
                "metrics": {"SWA": [], "CWA": [], "CompWA": []},
                "predictions": [],
                "ground_truth": [],
            },
        }
    }
}

save_slot = experiment_data["last_token_repr"]["SPR"]

# --------------------------------------------------------------------- paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------ dataset
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_synthetic_dataset(n_tr=2000, n_dev=500, n_test=500, max_len=10):
    def _row():
        L = random.randint(4, max_len)
        seq, label = [], 0
        for _ in range(L):
            sh, co = random.choice("ABCDE"), random.choice("01234")
            seq.append(sh + co)
            label ^= (ord(sh) + int(co)) & 1
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": label,
        }

    def _many(n):
        return [_row() for _ in range(n)]

    return DatasetDict(
        train=HFDataset.from_list(_many(n_tr)),
        dev=HFDataset.from_list(_many(n_dev)),
        test=HFDataset.from_list(_many(n_test)),
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_synthetic_dataset()
print({k: len(v) for k, v in spr.items()})

# ----------------------------------------------------------- vocab & encoding
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]
MAX_LEN = 40


def encode(seq, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.split()][:max_len]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# ------------------------------------------------------------------ metrics
def count_shape_variety(sequence):
    return len({tok[0] for tok in sequence.split()})


def count_color_variety(sequence):
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / sum(w) if sum(w) else 0.0
    )


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / sum(w) if sum(w) else 0.0
    )


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return (
        sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / sum(w) if sum(w) else 0.0
    )


# ------------------------------------------------------------ augmentations
def shape_rename(seq):
    toks = seq.split()
    mapping = {s: random.choice(string.ascii_uppercase) for s in {t[0] for t in toks}}
    return " ".join([mapping[t[0]] + t[1:] for t in toks])


def color_rename(seq):
    toks = seq.split()
    mapping = {
        c: random.choice("0123456789") for c in {t[1] for t in toks if len(t) > 1}
    }
    return " ".join([t[0] + mapping.get(t[1], t[1]) for t in toks])


def token_dropout(seq, p=0.15):
    toks = [t for t in seq.split() if random.random() > p]
    return " ".join(toks if toks else seq.split())


def augment(seq):
    if random.random() < 0.4:
        seq = shape_rename(seq)
    if random.random() < 0.4:
        seq = color_rename(seq)
    if random.random() < 0.3:
        seq = token_dropout(seq)
    return seq


# ---------------------------------------------------------- torch datasets
class ContrastiveSPRDataset(TorchDataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[idx]["sequence"]
        v1, v2 = augment(s), augment(s)
        return (
            torch.tensor(encode(v1), dtype=torch.long),
            torch.tensor(encode(v2), dtype=torch.long),
        )


class ClassificationSPRDataset(TorchDataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        r = self.ds[idx]
        return (
            torch.tensor(encode(r["sequence"]), dtype=torch.long),
            torch.tensor(r["label"], dtype=torch.long),
            r["sequence"],
        )


def collate_contrastive(batch):
    v1 = torch.stack([b[0] for b in batch])
    v2 = torch.stack([b[1] for b in batch])
    return {"view1": v1, "view2": v2}


def collate_classification(batch):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    seqs = [b[2] for b in batch]
    return {"input_ids": ids, "labels": labels, "sequence": seqs}


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    """
    Ablation: Return last forward & backward hidden state (no masked mean pooling)
    """

    def __init__(self, vocab_sz, emb_dim=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)

    def forward(self, x):  # x: B,L
        emb = self.emb(x)  # B,L,E
        out, h = self.gru(emb)  # h: (2,B,hid)
        h_fwd = h[0]  # B,hid
        h_bwd = h[1]  # B,hid
        return torch.cat([h_fwd, h_bwd], dim=-1)  # B,2*hid


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, T=0.07):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N,D
    sim = torch.matmul(z, z.t()) / T  # 2N,2N
    mask = (~torch.eye(2 * N, dtype=torch.bool, device=z.device)).float()
    sim = sim - 1e9 * (1 - mask)  # remove self-sim
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])
    loss = nn.CrossEntropyLoss()(sim, labels)
    return loss


# --------------------------------------------------- contrastive pre-train
BATCH_C = 256
pre_epochs = 2
train_subset = spr["train"].shuffle(seed=0).select(range(min(5000, len(spr["train"]))))
c_loader = DataLoader(
    ContrastiveSPRDataset(train_subset),
    batch_size=BATCH_C,
    shuffle=True,
    collate_fn=collate_contrastive,
)

encoder = Encoder(len(vocab)).to(device)
projector = Projector(512).to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(projector.parameters()), lr=3e-3
)

print("\n--- Contrastive pre-training ---")
for ep in range(1, pre_epochs + 1):
    encoder.train()
    projector.train()
    running = 0.0
    for batch in c_loader:
        v1 = batch["view1"].to(device)
        v2 = batch["view2"].to(device)
        z1 = projector(encoder(v1))
        z2 = projector(encoder(v2))
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    avg = running / len(c_loader)
    save_slot["contrastive_pretrain"]["losses"].append((ep, avg))
    print(f"Pre-epoch {ep}: contrastive_loss = {avg:.4f}")


# ------------------------------------------------------ fine-tune classifier
class Classifier(nn.Module):
    def __init__(self, enc, num_cls=2):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(512, num_cls)

    def forward(self, x):
        rep = self.enc(x)
        return self.fc(rep)


FINE_EPOCHS = 5
BATCH_F = 256
train_loader = DataLoader(
    ClassificationSPRDataset(spr["train"]),
    batch_size=BATCH_F,
    shuffle=True,
    collate_fn=collate_classification,
)
dev_loader = DataLoader(
    ClassificationSPRDataset(spr["dev"]),
    batch_size=BATCH_F,
    shuffle=False,
    collate_fn=collate_classification,
)
model = Classifier(encoder).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print("\n--- Fine-tuning ---")
for ep in range(1, FINE_EPOCHS + 1):
    # train
    model.train()
    run_loss = 0.0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        opt.zero_grad()
        logits = model(ids)
        loss = crit(logits, lbl)
        loss.backward()
        opt.step()
        run_loss += loss.item()
    tr_loss = run_loss / len(train_loader)
    save_slot["fine_tune"]["losses"]["train"].append((ep, tr_loss))
    # val
    model.eval()
    val_loss, seqs, preds, gts = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            logits = model(ids)
            val_loss += crit(logits, lbl).item()
            p = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            gts.extend(batch["labels"].tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(dev_loader)
    SWA = shape_weighted_accuracy(seqs, gts, preds)
    CWA = color_weighted_accuracy(seqs, gts, preds)
    CompWA = complexity_weighted_accuracy(seqs, gts, preds)
    save_slot["fine_tune"]["losses"]["val"].append((ep, val_loss))
    save_slot["fine_tune"]["metrics"]["SWA"].append((ep, SWA))
    save_slot["fine_tune"]["metrics"]["CWA"].append((ep, CWA))
    save_slot["fine_tune"]["metrics"]["CompWA"].append((ep, CompWA))
    save_slot["fine_tune"]["predictions"].append((ep, preds))
    save_slot["fine_tune"]["ground_truth"].append((ep, gts))
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f}  SWA={SWA:.4f} "
        f"CWA={CWA:.4f}  CompWA={CompWA:.4f}"
    )

# -------------------------------------------------------------- save & done
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
