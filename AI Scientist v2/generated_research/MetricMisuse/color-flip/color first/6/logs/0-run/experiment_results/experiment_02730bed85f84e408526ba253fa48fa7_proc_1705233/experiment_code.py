# Late-Fusion Dual-LSTM ablation for SPR-BENCH
import os, pathlib, time, random
from typing import List
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# -------------------------------------------------------------------------
# basic setup
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# locate SPR_BENCH folder (add other candidate paths if needed)
for _p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if _p.exists():
        DATA_PATH = _p
        break
else:
    raise FileNotFoundError("Place SPR_BENCH folder next to this script.")


def load_split(csv_name: str):
    return load_dataset(
        "csv",
        data_files=str(DATA_PATH / csv_name),
        split="train",
        cache_dir=".cache_dsets",
    )


dsets = {
    "train": load_split("train.csv"),
    "dev": load_split("dev.csv"),
    "test": load_split("test.csv"),
}
print({k: len(v) for k, v in dsets.items()})


# -------------------------------------------------------------------------
# utilities for metrics
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
# build vocabularies
shapes, colors, labels = set(), set(), set()
for row in dsets["train"]:
    labels.add(row["label"])
    for tok in row["sequence"].split():
        if tok:
            shapes.add(tok[0])
            if len(tok) > 1:
                colors.add(tok[1])
shape2idx = {s: i + 1 for i, s in enumerate(sorted(shapes))}  # 0 PAD
color2idx = {c: i + 1 for i, c in enumerate(sorted(colors))}
lab2idx = {l: i for i, l in enumerate(sorted(labels))}
idx2lab = {i: l for l, i in lab2idx.items()}
print(f"Vocab: {len(shape2idx)} shapes, {len(color2idx)} colors, {len(lab2idx)} labels")


# -------------------------------------------------------------------------
# dataset / dataloader
class DualChannelSPR(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labs = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        shp = [shape2idx.get(tok[0], 0) for tok in self.seqs[idx].split()]
        col = [
            color2idx.get(tok[1], 0) if len(tok) > 1 else 0
            for tok in self.seqs[idx].split()
        ]
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
# late-fusion model
class LateFusionDualLSTM(nn.Module):
    def __init__(
        self, shape_vocab, color_vocab, n_lab, edim_shape=32, edim_color=8, hid=128
    ):
        super().__init__()
        self.emb_s = nn.Embedding(shape_vocab, edim_shape, padding_idx=0)
        self.emb_c = nn.Embedding(color_vocab, edim_color, padding_idx=0)
        self.lstm_s = nn.LSTM(edim_shape, hid, batch_first=True, bidirectional=True)
        self.lstm_c = nn.LSTM(edim_color, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(4 * hid, n_lab)  # 2*hid per stream

    def forward(self, shape_ids, color_ids, lens):
        # shape stream
        emb_s = self.emb_s(shape_ids)
        packed_s = nn.utils.rnn.pack_padded_sequence(
            emb_s, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out_s, _ = self.lstm_s(packed_s)
        out_s, _ = nn.utils.rnn.pad_packed_sequence(out_s, batch_first=True)
        mask = (shape_ids != 0).unsqueeze(-1)
        mean_s = (out_s * mask).sum(1) / lens.unsqueeze(1).type_as(out_s)
        # color stream
        emb_c = self.emb_c(color_ids)
        packed_c = nn.utils.rnn.pack_padded_sequence(
            emb_c, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out_c, _ = self.lstm_c(packed_c)
        out_c, _ = nn.utils.rnn.pad_packed_sequence(out_c, batch_first=True)
        mean_c = (out_c * mask).sum(1) / lens.unsqueeze(1).type_as(out_c)
        # fuse
        fused = torch.cat([mean_s, mean_c], dim=-1)
        return self.fc(fused)


model = LateFusionDualLSTM(len(shape2idx) + 1, len(color2idx) + 1, len(lab2idx)).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------------------------------
# experiment tracking dict
experiment_data = {
    "late_fusion_dual_lstm": {
        "dual_channel": {
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
    running_loss, seen = 0, 0
    for batch in train_loader():
        b = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(b["shape"], b["color"], b["len"])
        loss = criterion(logits, b["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * b["shape"].size(0)
        seen += b["shape"].size(0)
    tr_loss = running_loss / seen
    experiment_data["late_fusion_dual_lstm"]["dual_channel"]["losses"]["train"].append(
        (epoch, tr_loss)
    )

    val_loss, seqs, yt, yp = run_eval(dev_loader)
    experiment_data["late_fusion_dual_lstm"]["dual_channel"]["losses"]["val"].append(
        (epoch, val_loss)
    )
    vcwa, vswa, vpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
    experiment_data["late_fusion_dual_lstm"]["dual_channel"]["metrics"]["val"].append(
        (epoch, {"CWA": vcwa, "SWA": vswa, "PCWA": vpcwa})
    )
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA={vcwa:.4f} | SWA={vswa:.4f} | PCWA={vpcwa:.4f}"
    )

# -------------------------------------------------------------------------
# testing
test_loss, seqs, yt, yp = run_eval(test_loader)
tcwa, tswa, tpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
print(f"Test  CWA={tcwa:.4f} | SWA={tswa:.4f} | PCWA={tpcwa:.4f}")
experiment_data["late_fusion_dual_lstm"]["dual_channel"]["predictions"] = yp
experiment_data["late_fusion_dual_lstm"]["dual_channel"]["ground_truth"] = yt

# -------------------------------------------------------------------------
# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
