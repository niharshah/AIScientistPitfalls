import os, pathlib, time
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# -------------------------------------------------------------------------
# experiment dict & workspace
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "shared_embedding": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------------------------------
# locate data
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


dsets = {n: load_split(f"{n}.csv") for n in ["train", "dev", "test"]}
print({k: len(v) for k, v in dsets.items()})

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

union_syms = sorted(shapes | colors)
sym2idx = {s: i + 1 for i, s in enumerate(union_syms)}  # 0 is PAD
lab2idx = {l: i for i, l in enumerate(sorted(labels))}
idx2lab = {i: l for l, i in lab2idx.items()}
print(f"Union vocab size (pad+syms): {len(sym2idx)+1}")


# -------------------------------------------------------------------------
# datasets / loaders
class SharedEmbedSPR(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labs = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        shp_ids, col_ids = [], []
        for tok in self.seqs[idx].split():
            if not tok:
                continue
            shp_ids.append(sym2idx.get(tok[0], 0))
            if len(tok) > 1:
                col_ids.append(sym2idx.get(tok[1], 0))
            else:
                col_ids.append(0)  # missing colour
        return {
            "shape": torch.tensor(shp_ids, dtype=torch.long),
            "color": torch.tensor(col_ids, dtype=torch.long),
            "len": len(shp_ids),
            "label": lab2idx[self.labs[idx]],
            "raw": self.seqs[idx],
        }


def collate(batch: List[Dict]):
    mx = max(b["len"] for b in batch)
    shp = torch.zeros(len(batch), mx, dtype=torch.long)
    col = torch.zeros(len(batch), mx, dtype=torch.long)
    labs, lens, raws = [], [], []
    for i, b in enumerate(batch):
        shp[i, : b["len"]] = b["shape"]
        col[i, : b["len"]] = b["color"]
        labs.append(b["label"])
        lens.append(b["len"])
        raws.append(b["raw"])
    return {
        "shape": shp,
        "color": col,
        "len": torch.tensor(lens),
        "label": torch.tensor(labs),
        "raw": raws,
    }


BS_TRAIN = 128
train_loader = lambda: DataLoader(
    SharedEmbedSPR(dsets["train"]),
    batch_size=BS_TRAIN,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SharedEmbedSPR(dsets["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SharedEmbedSPR(dsets["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


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
# model
class SharedEmbedBiLSTM(nn.Module):
    def __init__(self, vocab_sz, edim=32, hid=128, n_lab=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, edim, padding_idx=0)
        self.lstm = nn.LSTM(2 * edim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, n_lab)

    def forward(self, shape_ids, color_ids, lens):
        x = torch.cat([self.emb(shape_ids), self.emb(color_ids)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (shape_ids != 0).unsqueeze(-1)
        mean = (out * mask).sum(1) / lens.unsqueeze(1).type_as(out)
        return self.fc(mean)


model = SharedEmbedBiLSTM(len(sym2idx) + 1, edim=32, hid=128, n_lab=len(lab2idx)).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------------------------------------------------------------
# evaluation helper
@torch.no_grad()
def run_eval(loader):
    model.eval()
    seqs, yt, yp = [], [], []
    totloss, n = 0.0, 0
    for batch in loader:
        b = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
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
# training
EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss, seen = 0.0, 0
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
        tot_loss += loss.item() * batch["shape"].size(0)
        seen += batch["shape"].size(0)
    tr_loss = tot_loss / seen
    experiment_data["shared_embedding"]["losses"]["train"].append((epoch, tr_loss))

    val_loss, seqs, yt, yp = run_eval(dev_loader)
    experiment_data["shared_embedding"]["losses"]["val"].append((epoch, val_loss))
    vcwa, vswa, vpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
    experiment_data["shared_embedding"]["metrics"]["val"].append(
        (epoch, {"CWA": vcwa, "SWA": vswa, "PCWA": vpcwa})
    )
    print(
        f"Epoch {epoch}: val_loss {val_loss:.4f} | CWA {vcwa:.4f} | SWA {vswa:.4f} | PCWA {vpcwa:.4f}"
    )

# -------------------------------------------------------------------------
# testing
test_loss, seqs, yt, yp = run_eval(test_loader)
tcwa, tswa, tpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
experiment_data["shared_embedding"]["predictions"] = yp
experiment_data["shared_embedding"]["ground_truth"] = yt
print(
    f"TEST: loss {test_loss:.4f} | CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}"
)

# -------------------------------------------------------------------------
# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
