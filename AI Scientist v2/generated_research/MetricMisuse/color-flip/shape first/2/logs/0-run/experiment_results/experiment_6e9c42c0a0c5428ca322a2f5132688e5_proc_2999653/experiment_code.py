import os, pathlib, random, math, time
from typing import Dict, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- experiment store ---------------
experiment_data = {
    "joint_train": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ----------------- device ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------- reproducibility ---------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------- data paths ---------------------
def find_spr() -> pathlib.Path:
    cands = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in cands:
        p = pathlib.Path(c).expanduser()
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr()
print("SPR_BENCH folder:", DATA_PATH)


# ------------- helpers ------------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _one(csv):
        return load_dataset(
            "csv",
            data_files=str(root / csv),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    return DatasetDict(
        {"train": _one("train.csv"), "dev": _one("dev.csv"), "test": _one("test.csv")}
    )


def count_shape(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def swa(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def cwa(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def ccwa(seqs, y_t, y_p):
    w = [count_shape(s) + count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


spr = load_spr(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------- vocab + labels -----------------
def build_vocab(ds) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in ds:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(ds):
    labels = sorted({ex["label"] for ex in ds})
    return {l: i for i, l in enumerate(labels)}


vocab = build_vocab(spr["train"])
label2id = build_label_map(spr["train"])
id2label = {i: l for l, i in label2id.items()}
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print("vocab", len(vocab), "labels", num_labels)


# -------------- augmentation -----------------
def augment(seq: str) -> str:
    toks = seq.split()
    # random token masking
    toks = [tok if random.random() > 0.3 else "<unk>" for tok in toks]
    # small local shuffle
    if len(toks) > 1 and random.random() < 0.3:
        i = random.randrange(len(toks) - 1)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
    return " ".join(toks)


# -------------- datasets ---------------------
class SPRJointDS(torch.utils.data.Dataset):
    def __init__(self, hfds, vocab, label2id):
        self.data = hfds
        self.v = vocab
        self.l2i = label2id

    def enc(self, s):
        return [self.v.get(t, self.v["<unk>"]) for t in s.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        seq = ex["sequence"]
        return {
            "input_ids": torch.tensor(self.enc(seq), dtype=torch.long),
            "aug_ids": torch.tensor(self.enc(augment(seq)), dtype=torch.long),
            "label": torch.tensor(self.l2i[ex["label"]], dtype=torch.long),
            "sequence": seq,
        }


def collate_joint(batch):
    def pad(seqs):
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out

    return {
        "input_ids": pad([b["input_ids"] for b in batch]),
        "aug_ids": pad([b["aug_ids"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "sequences": [b["sequence"] for b in batch],
    }


train_ds = SPRJointDS(spr["train"], vocab, label2id)
dev_ds = SPRJointDS(spr["dev"], vocab, label2id)


# -------------- model ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,L,D)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SeqEncoder(nn.Module):
    def __init__(self, vocab, emb_dim=128, n_heads=4, n_layers=2, ff=256, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb_dim, padding_idx=pad_idx)
        self.pos = PositionalEncoding(emb_dim, 512)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff, batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.pad_idx = pad_idx

    def forward(self, x):
        mask = x == self.pad_idx
        h = self.emb(x)
        h = self.pos(h)
        h = self.tr(h, src_key_padding_mask=mask)
        # mean pooling over non-pad tokens
        lens = (~mask).sum(1).clamp(min=1)
        h = (h * ~mask.unsqueeze(-1)).sum(1) / lens.unsqueeze(-1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(encoder.proj[-1].out_features, num_labels)

    def forward(self, x):
        return self.head(self.enc(x))


# -------------- contrastive loss --------------
def nt_xent(z, temp=0.5):
    z = nn.functional.normalize(z, dim=1)
    B = z.size(0) // 2
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    labels = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, labels)


# ------------- training loop -----------------
def train_joint(epochs=8, batch_size=128, lr=1e-3, alpha=0.1, patience=3):
    encoder = SeqEncoder(vocab).to(device)
    model = Classifier(encoder, num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_joint,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=256, shuffle=False, collate_fn=collate_joint, num_workers=0
    )
    best_ccwa = -1
    no_imp = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t_loss = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss_ce = ce_loss(logits, batch["labels"])
            z_orig = model.enc(batch["input_ids"])
            z_aug = model.enc(batch["aug_ids"])
            loss_con = nt_xent(torch.cat([z_orig, z_aug], 0))
            loss = loss_ce + alpha * loss_con
            loss.backward()
            opt.step()
            t_loss += loss.item() * batch["labels"].size(0)
        t_loss /= len(train_ds)
        experiment_data["joint_train"]["losses"]["train"].append(t_loss)
        # ---- eval ----
        model.eval()
        d_loss = 0
        all_p = []
        all_t = []
        all_s = []
        with torch.no_grad():
            for batch in dev_loader:
                batch_t = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch_t["input_ids"])
                loss = ce_loss(logits, batch_t["labels"])
                d_loss += loss.item() * batch_t["labels"].size(0)
                preds = logits.argmax(-1).cpu().tolist()
                truths = batch_t["labels"].cpu().tolist()
                all_p.extend(preds)
                all_t.extend(truths)
                all_s.extend(batch["sequences"])
        d_loss /= len(dev_ds)
        swa_v = swa(all_s, all_t, all_p)
        cwa_v = cwa(all_s, all_t, all_p)
        ccwa_v = ccwa(all_s, all_t, all_p)
        experiment_data["joint_train"]["losses"]["val"].append(d_loss)
        experiment_data["joint_train"]["metrics"]["val"].append(
            {"epoch": epoch, "swa": swa_v, "cwa": cwa_v, "ccwa": ccwa_v, "loss": d_loss}
        )
        experiment_data["joint_train"]["metrics"]["train"].append(
            {"epoch": epoch, "loss": t_loss}
        )
        print(
            f"Epoch {epoch}: val_loss={d_loss:.4f} SWA={swa_v:.4f} CWA={cwa_v:.4f} CCWA={ccwa_v:.4f}"
        )
        # early stopping
        if ccwa_v > best_ccwa + 1e-6:
            best_ccwa = ccwa_v
            no_imp = 0
            best_state = model.state_dict()
            experiment_data["joint_train"]["predictions"] = all_p
            experiment_data["joint_train"]["ground_truth"] = all_t
        else:
            no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


# ------------- run experiment ----------------
model = train_joint()

# ------------- save metrics ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Metrics saved to", os.path.join(working_dir, "experiment_data.npy"))
