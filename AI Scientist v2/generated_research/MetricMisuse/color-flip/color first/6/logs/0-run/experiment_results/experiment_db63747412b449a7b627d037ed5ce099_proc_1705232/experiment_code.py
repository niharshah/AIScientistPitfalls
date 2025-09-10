# shape_only_ablation.py
import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# -------------------------------------------------------------------------
# I/O and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# locate data folder
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


dsets = {k: load_split(f"{k}.csv") for k in ["train", "dev", "test"]}
print({k: len(v) for k, v in dsets.items()})


# -------------------------------------------------------------------------
# metrics helpers (unchanged)
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
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
# vocabularies
shapes, colors, labels = set(), set(), set()
for r in dsets["train"]:
    labels.add(r["label"])
    for tok in r["sequence"].split():
        if tok:
            shapes.add(tok[0])
            if len(tok) > 1:
                colors.add(tok[1])
shape2idx = {s: i + 1 for i, s in enumerate(sorted(shapes))}  # 0 PAD
lab2idx = {l: i for i, l in enumerate(sorted(labels))}
idx2lab = {i: l for l, i in lab2idx.items()}
print(f"Vocab: {len(shape2idx)} shapes")


# -------------------------------------------------------------------------
# dataset & dataloader (reuse color parsing but will ignore color later)
class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labs = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        shp = [shape2idx.get(tok[0], 0) for tok in self.seqs[idx].split()]
        return {
            "shape": torch.tensor(shp, dtype=torch.long),
            "len": len(shp),
            "label": lab2idx[self.labs[idx]],
            "raw": self.seqs[idx],
        }


def collate(batch):
    mx = max(b["len"] for b in batch)
    shp = torch.zeros(len(batch), mx, dtype=torch.long)
    ln, lab, raw = [], [], []
    for i, b in enumerate(batch):
        shp[i, : b["len"]] = b["shape"]
        ln.append(b["len"])
        lab.append(b["label"])
        raw.append(b["raw"])
    return {
        "shape": shp,
        "len": torch.tensor(ln),
        "label": torch.tensor(lab),
        "raw": raw,
    }


BS_TRAIN = 128
train_loader = lambda: DataLoader(
    SPRDataset(dsets["train"]), batch_size=BS_TRAIN, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dsets["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(dsets["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# -------------------------------------------------------------------------
# Shape-only model
class ShapeOnlyBiLSTM(nn.Module):
    def __init__(self, shape_vocab, n_lab, edim=32, hid=128):
        super().__init__()
        self.emb = nn.Embedding(shape_vocab, edim, padding_idx=0)
        self.lstm = nn.LSTM(edim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, n_lab)

    def forward(self, shape_ids, lens):
        x = self.emb(shape_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (shape_ids != 0).unsqueeze(-1)
        mean = (out * mask).sum(1) / lens.unsqueeze(1).type_as(out)
        return self.fc(mean)


model = ShapeOnlyBiLSTM(len(shape2idx) + 1, len(lab2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------------------------------
# experiment tracking
experiment_data = {
    "shape_only": {
        "SPR": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
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
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(b["shape"], b["len"])
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
for ep in range(1, EPOCHS + 1):
    model.train()
    run_loss, seen = 0, 0
    for batch in train_loader():
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["shape"], batch["len"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["shape"].size(0)
        seen += batch["shape"].size(0)
    tr_loss = run_loss / seen
    experiment_data["shape_only"]["SPR"]["losses"]["train"].append((ep, tr_loss))

    val_loss, seqs, yt, yp = run_eval(dev_loader)
    experiment_data["shape_only"]["SPR"]["losses"]["val"].append((ep, val_loss))
    vcwa, vswa, vpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
    experiment_data["shape_only"]["SPR"]["metrics"]["val"].append(
        (ep, {"CWA": vcwa, "SWA": vswa, "PCWA": vpcwa})
    )
    print(
        f"Epoch {ep}: val_loss {val_loss:.4f} | CWA {vcwa:.4f} | SWA {vswa:.4f} | PCWA {vpcwa:.4f}"
    )

# -------------------------------------------------------------------------
# test evaluation
test_loss, seqs, yt, yp = run_eval(test_loader)
tcwa, tswa, tpcwa = cwa(seqs, yt, yp), swa(seqs, yt, yp), pcwa(seqs, yt, yp)
experiment_data["shape_only"]["SPR"]["predictions"] = yp
experiment_data["shape_only"]["SPR"]["ground_truth"] = yt
print(f"TEST  CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}")

# -------------------------------------------------------------------------
# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
