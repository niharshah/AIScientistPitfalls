import os, pathlib, time
from typing import List
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# -------------------------------------------------------------------------
# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# locate SPR_BENCH folder
for _p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if _p.exists():
        DATA_PATH = _p
        break
else:
    raise FileNotFoundError("Place SPR_BENCH folder next to this script.")


def load_split(name):
    return load_dataset(
        "csv",
        data_files=str(DATA_PATH / name),
        split="train",
        cache_dir=".cache_dsets",
    )


dsets = {k: load_split(f"{k}.csv") for k in ["train", "dev", "test"]}
print({k: len(v) for k, v in dsets.items()})


# -------------------------------------------------------------------------
# metric helpers
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def pcwa(seqs, y_t, y_p):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# -------------------------------------------------------------------------
# build vocabularies
shapes, colors, labels = set(), set(), set()
for row in dsets["train"]:
    labels.add(row["label"])
    for tok in row["sequence"].split():
        if tok:
            shapes.add(tok[0])
            if len(tok) > 1:
                colors.add(tok[1])
shape2idx = {s: i + 1 for i, s in enumerate(sorted(shapes))}  # 0 pad
color2idx = {c: i + 1 for i, c in enumerate(sorted(colors))}
lab2idx = {l: i for i, l in enumerate(sorted(labels))}
idx2lab = {i: l for l, i in lab2idx.items()}
print(f"Vocab sizes -> shapes:{len(shape2idx)}  colors:{len(color2idx)}")


# -------------------------------------------------------------------------
# dataset
class DualChannelSPR(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labs = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        shp = [shape2idx.get(t[0], 0) for t in toks]
        col = [color2idx.get(t[1], 0) if len(t) > 1 else 0 for t in toks]
        return {
            "shape": torch.tensor(shp, dtype=torch.long),
            "color": torch.tensor(col, dtype=torch.long),
            "len": len(shp),
            "label": lab2idx[self.labs[idx]],
            "raw": self.seqs[idx],
        }


def collate(batch):
    mx = max(b["len"] for b in batch)
    shp = torch.zeros(len(batch), mx, dtype=torch.long)
    col = torch.zeros(len(batch), mx, dtype=torch.long)
    lab, ln, raws = [], [], []
    for i, b in enumerate(batch):
        shp[i, : b["len"]] = b["shape"]
        col[i, : b["len"]] = b["color"]
        lab.append(b["label"])
        ln.append(b["len"])
        raws.append(b["raw"])
    return {
        "shape": shp,
        "color": col,
        "len": torch.tensor(ln),
        "label": torch.tensor(lab),
        "raw": raws,
    }


BS_TRAIN = 128
train_loader = lambda: DataLoader(
    DualChannelSPR(dsets["train"]),
    batch_size=BS_TRAIN,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    DualChannelSPR(dsets["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    DualChannelSPR(dsets["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# -------------------------------------------------------------------------
# Mean-Bag model (no LSTM)
class MeanBagEmbed(nn.Module):
    def __init__(self, shape_vocab, color_vocab, n_lab, edim_s=32, edim_c=8, hid=128):
        super().__init__()
        self.emb_s = nn.Embedding(shape_vocab, edim_s, padding_idx=0)
        self.emb_c = nn.Embedding(color_vocab, edim_c, padding_idx=0)
        self.fc1 = nn.Linear(edim_s + edim_c, hid)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid, n_lab)

    def forward(self, shape_ids, color_ids, lens):
        x = torch.cat([self.emb_s(shape_ids), self.emb_c(color_ids)], dim=-1)  # B,T,D
        mask = (shape_ids != 0).unsqueeze(-1)
        summed = (x * mask).sum(1)  # B,D
        mean = summed / lens.unsqueeze(1).type_as(x)  # B,D
        h = self.act(self.fc1(mean))
        return self.fc2(h)


model = MeanBagEmbed(len(shape2idx) + 1, len(color2idx) + 1, len(lab2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------------------------------
# experiment data dict
experiment_data = {
    "mean_bag": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------------------------------------------------------------
# evaluation helper
def run_eval(loader):
    model.eval()
    seqs, yt, yp = [], [], []
    totloss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(b["shape"], b["color"], b["len"])
            loss = criterion(logits, b["label"])
            totloss += loss.item() * b["shape"].size(0)
            n += b["shape"].size(0)
            pr = logits.argmax(1).cpu().tolist()
            tr = batch["label"].cpu().tolist()
            yp.extend([idx2lab[i] for i in pr])
            yt.extend([idx2lab[i] for i in tr])
            seqs.extend(batch["raw"])
    return totloss / n, seqs, yt, yp


# -------------------------------------------------------------------------
# training loop
EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, seen = 0, 0
    for batch in train_loader():
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["shape"], batch["color"], batch["len"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["shape"].size(0)
        seen += batch["shape"].size(0)
    tr_loss = run_loss / seen
    experiment_data["mean_bag"]["losses"]["train"].append((epoch, tr_loss))

    val_loss, seqs, yt, yp = run_eval(dev_loader)
    experiment_data["mean_bag"]["losses"]["val"].append((epoch, val_loss))
    vcwa, vswa, vpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
    experiment_data["mean_bag"]["metrics"]["val"].append(
        (epoch, {"CWA": vcwa, "SWA": vswa, "PCWA": vpcwa})
    )
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA {vcwa:.4f} | SWA {vswa:.4f} | PCWA {vpcwa:.4f}"
    )

# -------------------------------------------------------------------------
# testing
test_loss, seqs, yt, yp = run_eval(test_loader)
tcwa, tswa, tpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
experiment_data["mean_bag"]["predictions"] = yp
experiment_data["mean_bag"]["ground_truth"] = yt
print(
    f"TEST: loss={test_loss:.4f} | CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}"
)

# -------------------------------------------------------------------------
# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
