# -------------------------------------------------------------------------
# Unidirectional-LSTM ablation for SPR-Bench
# -------------------------------------------------------------------------
import os, pathlib, time, numpy as np
from typing import List
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# reproducibility (optional, comment out if not needed)
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------------------------------
# working dir & device
WORK_DIR = os.path.join(os.getcwd(), "working")
os.makedirs(WORK_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# -------------------------------------------------------------------------
# find SPR_BENCH data folder
for cand in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if cand.exists():
        DATA_PATH = cand
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found next to script.")


def load_split(csv_name: str):
    return load_dataset(
        "csv",
        data_files=str(DATA_PATH / csv_name),
        split="train",
        cache_dir=".cache_dsets",
    )


dsets = {k: load_split(f"{k}.csv") for k in ["train", "dev", "test"]}
print("Split sizes:", {k: len(v) for k, v in dsets.items()})


# -------------------------------------------------------------------------
# metrics helpers
def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


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
# build vocabularies (from training split only)
shapes, colors, labels = set(), set(), set()
for r in dsets["train"]:
    labels.add(r["label"])
    for tok in r["sequence"].split():
        if tok:
            shapes.add(tok[0])
            if len(tok) > 1:
                colors.add(tok[1])
shape2idx = {s: i + 1 for i, s in enumerate(sorted(shapes))}  # 0 reserved for PAD
color2idx = {c: i + 1 for i, c in enumerate(sorted(colors))}
lab2idx = {l: i for i, l in enumerate(sorted(labels))}
idx2lab = {i: l for l, i in lab2idx.items()}
print(f"Vocab: {len(shape2idx)} shapes, {len(color2idx)} colors, {len(lab2idx)} labels")


# -------------------------------------------------------------------------
# dataset / dataloader utilities
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


def collate(batch: List[dict]):
    max_len = max(b["len"] for b in batch)
    shp = torch.zeros(len(batch), max_len, dtype=torch.long)
    col = torch.zeros_like(shp)
    lab, lens, raws = [], [], []
    for i, b in enumerate(batch):
        l = b["len"]
        shp[i, :l] = b["shape"]
        col[i, :l] = b["color"]
        lab.append(b["label"])
        lens.append(l)
        raws.append(b["raw"])
    return {
        "shape": shp,
        "color": col,
        "len": torch.tensor(lens),
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
# Unidirectional LSTM model (hidden size halved wrt baseline 128 -> 64)
class DualEmbedUniLSTM(nn.Module):
    def __init__(
        self, shape_vocab, color_vocab, n_lab, edim_shape=32, edim_color=8, hid=64
    ):  # hid=64 (half)
        super().__init__()
        self.emb_s = nn.Embedding(shape_vocab, edim_shape, padding_idx=0)
        self.emb_c = nn.Embedding(color_vocab, edim_color, padding_idx=0)
        self.lstm = nn.LSTM(
            edim_shape + edim_color, hid, batch_first=True, bidirectional=False
        )
        self.fc = nn.Linear(hid, n_lab)

    def forward(self, shape_ids, color_ids, lens):
        x = torch.cat([self.emb_s(shape_ids), self.emb_c(color_ids)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (shape_ids != 0).unsqueeze(-1)
        mean = (out * mask).sum(1) / lens.unsqueeze(1).type_as(out)
        return self.fc(mean)


model = DualEmbedUniLSTM(len(shape2idx) + 1, len(color2idx) + 1, len(lab2idx)).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------------------------------
# experiment tracking dict
experiment_data = {
    "uni_lstm": {
        "spr": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# -------------------------------------------------------------------------
# helpers
def run_eval(loader):
    model.eval()
    seqs, yt, yp = [], [], []
    totloss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["shape"], batch["color"], batch["len"])
            loss = criterion(logits, batch["label"])
            totloss += loss.item() * batch["shape"].size(0)
            n += batch["shape"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            gold = batch["label"].cpu().tolist()
            yp.extend([idx2lab[i] for i in preds])
            yt.extend([idx2lab[i] for i in gold])
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
            k: v.to(device) if isinstance(v, torch.Tensor) else v
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
    experiment_data["uni_lstm"]["spr"]["losses"]["train"].append((epoch, tr_loss))

    # validation
    val_loss, seqs, yt, yp = run_eval(dev_loader)
    experiment_data["uni_lstm"]["spr"]["losses"]["val"].append((epoch, val_loss))
    vcwa, vswa, vpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
    experiment_data["uni_lstm"]["spr"]["metrics"]["val"].append(
        (epoch, {"CWA": vcwa, "SWA": vswa, "PCWA": vpcwa})
    )
    print(
        f"Epoch {epoch:02d} | val_loss {val_loss:.4f} | CWA {vcwa:.4f} | SWA {vswa:.4f} | PCWA {vpcwa:.4f}"
    )

# -------------------------------------------------------------------------
# final test evaluation
test_loss, seqs, yt, yp = run_eval(test_loader)
tcwa, tswa, tpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
print(
    f"TEST | loss {test_loss:.4f} | CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}"
)

experiment_data["uni_lstm"]["spr"]["predictions"] = yp
experiment_data["uni_lstm"]["spr"]["ground_truth"] = yt
experiment_data["uni_lstm"]["spr"]["metrics"]["test"] = {
    "CWA": tcwa,
    "SWA": tswa,
    "PCWA": tpcwa,
}

# -------------------------------------------------------------------------
# save
np.save(os.path.join(WORK_DIR, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", WORK_DIR)
