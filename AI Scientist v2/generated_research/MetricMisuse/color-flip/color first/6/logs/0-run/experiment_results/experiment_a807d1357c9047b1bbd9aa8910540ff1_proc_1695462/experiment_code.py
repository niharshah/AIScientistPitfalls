import os, pathlib, random, math, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ---------------- mandatory work dir --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device handling -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- reproducibility -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ---------------- SPR loader (reuse utility) ------------------
def load_spr_bench(root: pathlib.Path):
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH directory not found.")

dset = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dset.items()})


# ---------------- metrics -------------------------------------
def count_color_variety(seq: str) -> int:
    return len({tok[1:] if len(tok) > 1 else "" for tok in seq.split()})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] if tok else "" for tok in seq.split()})


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0


def pattern_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0


# ---------------- vocab construction --------------------------
def split_token(tok: str):
    if not tok:
        return "?", "?"
    shape = tok[0]
    color = tok[1:] if len(tok) > 1 else "?"
    return shape, color


shapes = set()
colors = set()
for s in dset["train"]["sequence"]:
    for t in s.split():
        sh, co = split_token(t)
        shapes.add(sh)
        colors.add(co)
shape2idx = {sh: i + 2 for i, sh in enumerate(sorted(shapes))}  # +2 reserve pad/unk
shape2idx["<pad>"] = 0
shape2idx["<unk>"] = 1
color2idx = {co: i + 2 for i, co in enumerate(sorted(colors))}
color2idx["<pad>"] = 0
color2idx["<unk>"] = 1
n_shape = len(shape2idx)
n_color = len(color2idx)
print(f"Shapes={n_shape-2}, Colors={n_color-2}")

labels = sorted(set(dset["train"]["label"]))
lab2idx = {l: i for i, l in enumerate(labels)}
idx2lab = {i: l for l, i in lab2idx.items()}
n_labels = len(labels)


# ---------------- Dataset & Dataloader ------------------------
class SPRGlyphDS(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lab = split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        toks = self.seq[idx].split()
        shape_ids = [shape2idx.get(split_token(t)[0], shape2idx["<unk>"]) for t in toks]
        color_ids = [color2idx.get(split_token(t)[1], color2idx["<unk>"]) for t in toks]
        return {
            "shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "color_ids": torch.tensor(color_ids, dtype=torch.long),
            "length": len(toks),
            "label": lab2idx[self.lab[idx]],
            "raw": self.seq[idx],
        }


def collate(batch):
    max_len = max(b["length"] for b in batch)
    sh_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    co_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    lens, labs, raws = [], [], []
    for i, b in enumerate(batch):
        L = b["length"]
        sh_pad[i, :L] = b["shape_ids"]
        co_pad[i, :L] = b["color_ids"]
        lens.append(L)
        labs.append(b["label"])
        raws.append(b["raw"])
    return {
        "shape_ids": sh_pad,
        "color_ids": co_pad,
        "lengths": torch.tensor(lens),
        "labels": torch.tensor(labs),
        "raw": raws,
    }


train_loader = DataLoader(
    SPRGlyphDS(dset["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRGlyphDS(dset["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRGlyphDS(dset["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------------- Model ---------------------------------------
class ShapeColorMean(nn.Module):
    def __init__(self, n_shape, n_color, d_emb, n_labels):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, d_emb, padding_idx=0)
        self.color_emb = nn.Embedding(n_color, d_emb, padding_idx=0)
        self.fc = nn.Linear(d_emb, n_labels)

    def forward(self, sh, co, lens):
        e = self.shape_emb(sh) + self.color_emb(co)
        mask = (sh != 0).unsqueeze(-1)
        summed = (e * mask).sum(1)
        mean = summed / lens.unsqueeze(1).clamp(min=1).float()
        return self.fc(mean)


model = ShapeColorMean(n_shape, n_color, 64, n_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# ---------------- experiment data dict -----------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- training loop ------------------------------
EPOCHS = 7


def evaluate(dloader):
    model.eval()
    tot_loss, n = 0, 0
    seqs, gts, prs = [], [], []
    with torch.no_grad():
        for batch in dloader:
            sh = batch["shape_ids"].to(device)
            co = batch["color_ids"].to(device)
            lens = batch["lengths"].to(device)
            labs = batch["labels"].to(device)
            logits = model(sh, co, lens)
            loss = criterion(logits, labs)
            tot_loss += loss.item() * sh.size(0)
            n += sh.size(0)
            preds = logits.argmax(1).cpu().tolist()
            labs_cpu = labs.cpu().tolist()
            seqs.extend(batch["raw"])
            gts.extend([idx2lab[i] for i in labs_cpu])
            prs.extend([idx2lab[i] for i in preds])
    val_loss = tot_loss / n
    cwa = color_weighted_accuracy(seqs, gts, prs)
    swa = shape_weighted_accuracy(seqs, gts, prs)
    pcwa = pattern_complexity_weighted_accuracy(seqs, gts, prs)
    return val_loss, cwa, swa, pcwa, gts, prs, seqs


for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    tot_loss = 0
    n = 0
    for batch in train_loader:
        sh = batch["shape_ids"].to(device)
        co = batch["color_ids"].to(device)
        lens = batch["lengths"].to(device)
        labs = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(sh, co, lens), labs)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * sh.size(0)
        n += sh.size(0)
    train_loss = tot_loss / n
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))

    # ---- validation ----
    val_loss, cwa, swa, pcwa, _, _, _ = evaluate(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (epoch, {"CWA": cwa, "SWA": swa, "PCWA": pcwa})
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA {cwa:.4f} | SWA {swa:.4f} | PCWA {pcwa:.4f} | epoch_time {time.time()-t0:.1f}s"
    )

# ---------------- final test ---------------------------------
test_loss, cwa, swa, pcwa, gts, prs, seqs = evaluate(test_loader)
experiment_data["SPR_BENCH"]["predictions"] = prs
experiment_data["SPR_BENCH"]["ground_truth"] = gts
print(
    f"\n=== Test ===  loss {test_loss:.4f} | CWA {cwa:.4f} | SWA {swa:.4f} | PCWA {pcwa:.4f}"
)

# ---------------- save ---------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
