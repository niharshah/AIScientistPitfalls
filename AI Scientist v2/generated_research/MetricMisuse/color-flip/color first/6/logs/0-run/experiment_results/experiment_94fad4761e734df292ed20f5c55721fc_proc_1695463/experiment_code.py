import os, pathlib, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR_BENCH (csv) ----------
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


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- build vocabularies ----------
shapes = sorted(
    {tok[0] for seq in spr["train"]["sequence"] for tok in seq.split() if tok}
)
colors = sorted(
    {tok[1] for seq in spr["train"]["sequence"] for tok in seq.split() if len(tok) > 1}
)

shape2i = {s: i + 1 for i, s in enumerate(shapes)}  # 0 is PAD
color2i = {c: i + 1 for i, c in enumerate(colors)}
v_shapes, v_colors = len(shape2i) + 1, len(color2i) + 1

labels = sorted(set(spr["train"]["label"]))
lab2i = {l: i for i, l in enumerate(labels)}
i2lab = {i: l for l, i in lab2i.items()}


# ---------- metrics ----------
def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def cwa(S, y, yh):
    w = [count_color_variety(s) for s in S]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y, yh)]
    return sum(c) / sum(w)


def swa(S, y, yh):
    w = [count_shape_variety(s) for s in S]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y, yh)]
    return sum(c) / sum(w)


def pcwa(S, y, yh):
    w = [count_color_variety(s) + count_shape_variety(s) for s in S]
    c = [wt if a == b else 0 for wt, a, b in zip(w, y, yh)]
    return sum(c) / sum(w)


# ---------- dataset ----------
class SPRDataset(Dataset):
    def __init__(self, hf):
        self.seqs = hf["sequence"]
        self.labs = hf["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        s_ids = [shape2i.get(t[0], 0) for t in toks]
        c_ids = [color2i.get(t[1], 0) if len(t) > 1 else 0 for t in toks]
        return {
            "shape": torch.tensor(s_ids, dtype=torch.long),
            "color": torch.tensor(c_ids, dtype=torch.long),
            "len": len(toks),
            "label": lab2i[self.labs[idx]],
            "raw": self.seqs[idx],
        }


def collate(batch):
    maxL = max(b["len"] for b in batch)
    shp = torch.zeros(len(batch), maxL, dtype=torch.long)
    col = torch.zeros_like(shp)
    lens, labels, raws = [], [], []
    for i, b in enumerate(batch):
        L = b["len"]
        shp[i, :L] = b["shape"]
        col[i, :L] = b["color"]
        lens.append(L)
        labels.append(b["label"])
        raws.append(b["raw"])
    return {
        "shape": shp,
        "color": col,
        "lens": torch.tensor(lens),
        "labels": torch.tensor(labels),
        "raw": raws,
    }


train_ds, dev_ds, test_ds = map(SPRDataset, (spr["train"], spr["dev"], spr["test"]))
train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

max_len = max(len(seq.split()) for seq in spr["train"]["sequence"]) + 1


# ---------- model ----------
class ShapeColorTransformer(nn.Module):
    def __init__(self, vs, vc, d_model, nhead, nlayers, n_lbl, max_len):
        super().__init__()
        self.shape_emb = nn.Embedding(vs, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(vc, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.fc = nn.Linear(d_model, n_lbl)

    def forward(self, shp, col, lens):
        B, S = shp.size()
        pos = torch.arange(S, device=shp.device).unsqueeze(0).expand(B, S)
        x = self.shape_emb(shp) + self.color_emb(col) + self.pos_emb(pos)
        x = x.transpose(0, 1)  # S,B,D
        mask = shp == 0  # B,S True->pad
        out = self.encoder(x, src_key_padding_mask=mask)
        out = out.transpose(0, 1)  # B,S,D
        m = (shp != 0).unsqueeze(-1)
        rep = (out * m).sum(1) / lens.unsqueeze(1).type_as(out)
        return self.fc(rep)


model = ShapeColorTransformer(v_shapes, v_colors, 64, 4, 2, len(lab2i), max_len).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- logging dict ----------
experiment_data = {
    "shape_color_transformer": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- helpers ----------
def evaluate(loader):
    model.eval()
    seqs, tgt, pred, tot_loss, n = [], [], [], 0, 0
    with torch.no_grad():
        for batch in loader:
            shp = batch["shape"].to(device)
            col = batch["color"].to(device)
            lens = batch["lens"].to(device)
            labs = batch["labels"].to(device)
            logits = model(shp, col, lens)
            loss = criterion(logits, labs)
            tot_loss += loss.item() * shp.size(0)
            n += shp.size(0)
            p = logits.argmax(1).cpu().tolist()
            t = labs.cpu().tolist()
            seqs.extend(batch["raw"])
            tgt.extend([i2lab[i] for i in t])
            pred.extend([i2lab[i] for i in p])
    return tot_loss / n, seqs, tgt, pred


# ---------- training ----------
EPOCHS = 5
BATCH = 128
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot, n = 0, 0
    for batch in train_loader(BATCH):
        shp = batch["shape"].to(device)
        col = batch["color"].to(device)
        lens = batch["lens"].to(device)
        labs = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(shp, col, lens), labs)
        loss.backward()
        optimizer.step()
        tot += loss.item() * shp.size(0)
        n += shp.size(0)
    tr_loss = tot / n
    experiment_data["shape_color_transformer"]["losses"]["train"].append(
        (epoch, tr_loss)
    )

    val_loss, seqs, y_true, y_pred = evaluate(dev_loader)
    experiment_data["shape_color_transformer"]["losses"]["val"].append(
        (epoch, val_loss)
    )
    cwa_v, swa_v, pcwa_v = (
        cwa(seqs, y_true, y_pred),
        swa(seqs, y_true, y_pred),
        pcwa(seqs, y_true, y_pred),
    )
    experiment_data["shape_color_transformer"]["metrics"]["val"].append(
        (epoch, {"CWA": cwa_v, "SWA": swa_v, "PCWA": pcwa_v})
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | CWA {cwa_v:.4f} | SWA {swa_v:.4f} | PCWA {pcwa_v:.4f}"
    )

# ---------- test ----------
_, seqs, y_true, y_pred = evaluate(test_loader)
experiment_data["shape_color_transformer"]["predictions"] = y_pred
experiment_data["shape_color_transformer"]["ground_truth"] = y_true
tcwa, tswa, tpcwa = (
    cwa(seqs, y_true, y_pred),
    swa(seqs, y_true, y_pred),
    pcwa(seqs, y_true, y_pred),
)
print(f"Test  CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
