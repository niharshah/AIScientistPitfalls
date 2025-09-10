import os, pathlib, math, time
from typing import List
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# --------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------
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


for cand in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]:
    if cand.exists():
        DATA_PATH = cand
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------------------- vocab building ------------------------------
def split_token(tok: str):
    tok = tok.strip()
    if not tok:
        return ("<pad>", "<pad>")
    if len(tok) == 1:
        return (tok[0], "<pad>")
    return (tok[0], tok[1])


shapes = sorted(
    {split_token(t)[0] for s in spr["train"]["sequence"] for t in s.split()}
)
colors = sorted(
    {split_token(t)[1] for s in spr["train"]["sequence"] for t in s.split()}
)
shape2idx = {s: i + 2 for i, s in enumerate(shapes)}  # 0 pad, 1 cls
color2idx = {c: i + 1 for i, c in enumerate(colors)}  # 0 pad
CLS_SHAPE_IDX = 1  # reserve idx 1 for CLS token shape, color uses 0
vocab_shape = len(shape2idx) + 2
vocab_color = len(color2idx) + 1
print(f"Shapes:{len(shape2idx)} Colors:{len(color2idx)}")

labels = sorted(set(spr["train"]["label"]))
lab2idx = {l: i for i, l in enumerate(labels)}
idx2lab = {i: l for l, i in lab2idx.items()}


# ---------------------- metrics -------------------------------------
def count_color_variety(seq: str) -> int:
    return len({split_token(tok)[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({split_token(tok)[0] for tok in seq.split() if tok})


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def pcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ----------------------- dataset ------------------------------------
class SPRDual(Dataset):
    def __init__(self, hf):
        self.seq = hf["sequence"]
        self.lab = hf["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        tokens = self.seq[idx].split()
        s_ids = [shape2idx.get(split_token(t)[0], 0) for t in tokens]
        c_ids = [color2idx.get(split_token(t)[1], 0) for t in tokens]
        return {
            "shape": torch.tensor(s_ids, dtype=torch.long),
            "color": torch.tensor(c_ids, dtype=torch.long),
            "len": len(tokens),
            "label": lab2idx[self.lab[idx]],
            "raw": self.seq[idx],
        }


def collate(batch):
    maxlen = max(b["len"] for b in batch) + 1  # +1 for CLS
    bs = len(batch)
    shp = torch.zeros(bs, maxlen, dtype=torch.long)
    col = torch.zeros(bs, maxlen, dtype=torch.long)
    lens = []
    labels = []
    raws = []
    for i, b in enumerate(batch):
        shp[i, 0] = CLS_SHAPE_IDX  # CLS at pos0
        col[i, 0] = 0
        l = b["len"]
        shp[i, 1 : l + 1] = b["shape"]
        col[i, 1 : l + 1] = b["color"]
        lens.append(l + 1)
        labels.append(b["label"])
        raws.append(b["raw"])
    return {
        "shape": shp,
        "color": col,
        "lens": torch.tensor(lens),
        "labels": torch.tensor(labels),
        "raw": raws,
    }


train_ds = SPRDual(spr["train"])
dev_ds = SPRDual(spr["dev"])
test_ds = SPRDual(spr["test"])
train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ----------------------- model --------------------------------------
class DualEmbTransformer(nn.Module):
    def __init__(self, vs_shape, vs_color, d_model=64, nhead=4, nlayers=2, n_lbl=10):
        super().__init__()
        self.shape_emb = nn.Embedding(vs_shape, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(vs_color, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(256, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.cls_fc = nn.Linear(d_model, n_lbl)

    def forward(self, shape, color, lens):
        pos_idx = torch.arange(shape.size(1), device=shape.device)
        pos = self.pos_emb(pos_idx)[None, :, :]
        tok_emb = self.shape_emb(shape) + self.color_emb(color) + pos
        key_padding = shape == 0
        enc = self.encoder(tok_emb, src_key_padding_mask=key_padding)
        # mean pool excluding pad
        mask = (~key_padding).unsqueeze(-1)
        pooled = (enc * mask).sum(1) / lens.unsqueeze(1).type_as(enc)
        return self.cls_fc(pooled)


model = DualEmbTransformer(vocab_shape, vocab_color, n_lbl=len(lab2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 6
BATCH = 128
experiment_data = {
    "dual_channel": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


def evaluate(loader):
    model.eval()
    tot_loss, n = 0, 0
    seqs, tru, pred = [], [], []
    with torch.no_grad():
        for b in loader:
            shp = b["shape"].to(device)
            col = b["color"].to(device)
            lens = b["lens"].to(device)
            labs = b["labels"].to(device)
            out = model(shp, col, lens)
            loss = criterion(out, labs)
            tot_loss += loss.item() * shp.size(0)
            n += shp.size(0)
            p = out.argmax(1).cpu().tolist()
            t = labs.cpu().tolist()
            seqs.extend(b["raw"])
            tru.extend([idx2lab[i] for i in t])
            pred.extend([idx2lab[i] for i in p])
    return tot_loss / n, seqs, tru, pred


for epoch in range(1, EPOCHS + 1):
    model.train()
    tot, n = 0, 0
    for b in train_loader(BATCH):
        shp = b["shape"].to(device)
        col = b["color"].to(device)
        lens = b["lens"].to(device)
        labs = b["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(shp, col, lens), labs)
        loss.backward()
        optimizer.step()
        tot += loss.item() * shp.size(0)
        n += shp.size(0)
    tr_loss = tot / n
    experiment_data["dual_channel"]["losses"]["train"].append((epoch, tr_loss))
    # validation
    val_loss, seqs, y_t, y_p = evaluate(dev_loader)
    experiment_data["dual_channel"]["losses"]["val"].append((epoch, val_loss))
    cwa_v, swa_v, pcwa_v = (
        cwa(seqs, y_t, y_p),
        swa(seqs, y_t, y_p),
        pcwa(seqs, y_t, y_p),
    )
    experiment_data["dual_channel"]["metrics"]["val"].append(
        (epoch, {"CWA": cwa_v, "SWA": swa_v, "PCWA": pcwa_v})
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
        f"CWA {cwa_v:.4f} | SWA {swa_v:.4f} | PCWA {pcwa_v:.4f}"
    )

# ----------------------- test ----------------------------------------
_, seqs, gt, pred = evaluate(test_loader)
experiment_data["dual_channel"]["predictions"] = pred
experiment_data["dual_channel"]["ground_truth"] = gt
tcwa, tswa, tpcwa = cwa(seqs, gt, pred), swa(seqs, gt, pred), pcwa(seqs, gt, pred)
print(f"Test  CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}")

# ----------------------- save ----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to working/experiment_data.npy")
