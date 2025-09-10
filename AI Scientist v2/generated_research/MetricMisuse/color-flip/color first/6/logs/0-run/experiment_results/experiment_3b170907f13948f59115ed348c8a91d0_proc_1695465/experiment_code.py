import os, math, random, pathlib, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset loading ----------
def load_spr_bench(root: pathlib.Path):
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


possible_paths = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
for p in possible_paths:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found.")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- helper: glyph splitting ----------
def split_glyph(g):
    if len(g) == 0:
        return "", ""
    if len(g) == 1:
        return g[0], ""
    return g[0], g[1]


# ---------- build separate vocabularies ----------
def build_vocab(tokens):
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


all_shapes, all_colors = [], []
for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        s, c = split_glyph(tok)
        all_shapes.append(s)
        all_colors.append(c)
shape_vocab = build_vocab(all_shapes)
color_vocab = build_vocab(all_colors)
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}

print(f"Shapes {len(shape_vocab)}, Colors {len(color_vocab)}, Labels {len(label2idx)}")


# ---------- Dataset ----------
class SPRDualDataset(Dataset):
    def __init__(self, hf_ds):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        shapes, colors = [], []
        for tok in self.seq[idx].strip().split():
            s, c = split_glyph(tok)
            shapes.append(shape_vocab.get(s, 1))
            colors.append(color_vocab.get(c, 1))
        return {
            "shape_ids": torch.tensor(shapes, dtype=torch.long),
            "color_ids": torch.tensor(colors, dtype=torch.long),
            "length": len(shapes),
            "label": label2idx[self.lab[idx]],
            "seq_raw": self.seq[idx],
        }


def collate(batch):
    max_len = max(b["length"] for b in batch)
    shp = torch.zeros(len(batch), max_len, dtype=torch.long)
    col = torch.zeros(len(batch), max_len, dtype=torch.long)
    lens, labs, raws = [], [], []
    for i, b in enumerate(batch):
        l = b["length"]
        shp[i, :l] = b["shape_ids"]
        col[i, :l] = b["color_ids"]
        lens.append(l)
        labs.append(b["label"])
        raws.append(b["seq_raw"])
    return {
        "shape_ids": shp,
        "color_ids": col,
        "lengths": torch.tensor(lens),
        "labels": torch.tensor(labs),
        "seq_raw": raws,
    }


train_ds, dev_ds, test_ds = map(SPRDualDataset, [spr["train"], spr["dev"], spr["test"]])
train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- metrics ----------
def count_color_var(sequence):
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


def count_shape_var(sequence):
    return len({tok[0] for tok in sequence.split() if tok})


def cwa(seq, y, p):
    w = [count_color_var(s) for s in seq]
    return sum(wi for wi, yt, pp in zip(w, y, p) if yt == pp) / max(sum(w), 1)


def swa(seq, y, p):
    w = [count_shape_var(s) for s in seq]
    return sum(wi for wi, yt, pp in zip(w, y, p) if yt == pp) / max(sum(w), 1)


def pcwa(seq, y, p):
    w = [count_color_var(s) + count_shape_var(s) for s in seq]
    return sum(wi for wi, yt, pp in zip(w, y, p) if yt == pp) / max(sum(w), 1)


# ---------- model ----------
class DualEmbedClassifier(nn.Module):
    def __init__(
        self, shape_vsz, color_vsz, d_shape=16, d_color=16, hidden=64, n_lab=10
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(shape_vsz, d_shape, padding_idx=0)
        self.color_emb = nn.Embedding(color_vsz, d_color, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(d_shape + d_color, hidden), nn.ReLU(), nn.Linear(hidden, n_lab)
        )

    def forward(self, shp_ids, col_ids, lens):
        mask = (shp_ids != 0).unsqueeze(-1)
        shp_mean = (self.shape_emb(shp_ids) * mask).sum(1) / lens.unsqueeze(1)
        col_mean = (self.color_emb(col_ids) * mask).sum(1) / lens.unsqueeze(1)
        feat = torch.cat([shp_mean, col_mean], dim=1)
        return self.mlp(feat)


# ---------- training setup ----------
EPOCHS = 8
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}
model = DualEmbedClassifier(
    len(shape_vocab), len(color_vocab), n_lab=len(label2idx)
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

# ---------- training loop ----------
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tot_loss, n = 0, 0
    for batch in train_loader(128):
        shp = batch["shape_ids"].to(device)
        col = batch["color_ids"].to(device)
        lens = batch["lengths"].to(device)
        labs = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(shp, col, lens)
        loss = criterion(logits, labs)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * shp.size(0)
        n += shp.size(0)
    tr_loss = tot_loss / n
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))
    # ---- dev ----
    model.eval()
    val_loss, n = 0, 0
    seqs, true, pred = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            shp = batch["shape_ids"].to(device)
            col = batch["color_ids"].to(device)
            lens = batch["lengths"].to(device)
            labs = batch["labels"].to(device)
            logits = model(shp, col, lens)
            loss = criterion(logits, labs)
            val_loss += loss.item() * shp.size(0)
            n += shp.size(0)
            pr = logits.argmax(1).cpu().tolist()
            la = labs.cpu().tolist()
            seqs.extend(batch["seq_raw"])
            true.extend([idx2label[i] for i in la])
            pred.extend([idx2label[i] for i in pr])
    val_loss /= n
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    cwa_s, swa_s, pcwa_s = (
        cwa(seqs, true, pred),
        swa(seqs, true, pred),
        pcwa(seqs, true, pred),
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (epoch, {"CWA": cwa_s, "SWA": swa_s, "PCWA": pcwa_s})
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA {cwa_s:.4f} | SWA {swa_s:.4f} | PCWA {pcwa_s:.4f}"
    )

# ---------- test evaluation ----------
model.eval()
seqs, true, pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        shp = batch["shape_ids"].to(device)
        col = batch["color_ids"].to(device)
        lens = batch["lengths"].to(device)
        logits = model(shp, col, lens)
        pr = logits.argmax(1).cpu().tolist()
        la = batch["labels"].cpu().tolist()
        seqs.extend(batch["seq_raw"])
        true.extend([idx2label[i] for i in la])
        pred.extend([idx2label[i] for i in pr])
tcwa, tswa, tpcwa = cwa(seqs, true, pred), swa(seqs, true, pred), pcwa(seqs, true, pred)
print(f"Test | CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = pred
experiment_data["SPR_BENCH"]["ground_truth"] = true

# ---------- save plots + data ----------
ep = list(range(1, EPOCHS + 1))
tr = [l for _, l in experiment_data["SPR_BENCH"]["losses"]["train"]]
vl = [l for _, l in experiment_data["SPR_BENCH"]["losses"]["val"]]
plt.figure()
plt.plot(ep, tr, label="train")
plt.plot(ep, vl, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curve")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
