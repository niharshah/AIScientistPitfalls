# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, math, time, random, string, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------- basic set-up ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- data loading ----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # tiny synthetic fallback
    print("SPR_BENCH missing: synthesising toy data")

    def synth(n):  # simple parity rule on “A”
        for i in range(n):
            seq = "".join(
                random.choices(string.ascii_uppercase[:12], k=random.randint(5, 15))
            )
            yield {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}

    spr = DatasetDict(
        {
            "train": load_dataset(
                "json", data_files={"train": list(synth(4000))}, split="train"
            ),
            "dev": load_dataset(
                "json", data_files={"train": list(synth(800))}, split="train"
            ),
            "test": load_dataset(
                "json", data_files={"train": list(synth(800))}, split="train"
            ),
        }
    )
print({k: len(v) for k, v in spr.items()})

# ---------------- vocab + encoding ------------
vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vsize = len(vocab)
MAX_LEN = min(max(len(ex["sequence"]) for ex in spr["train"]) + 1, 128)


def enc(seq):
    ids = [vocab["<cls>"]] + [vocab.get(c, 1) for c in seq][: MAX_LEN - 1]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


# estimated complexity = number of unique tokens (proxy if not provided)
def complexity(ex):
    return float(len(set(ex["sequence"])))


class SPRTorch(Dataset):
    def __init__(self, hf):
        self.d = hf

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        ex = self.d[idx]
        return {
            "input_ids": torch.tensor(enc(ex["sequence"]), dtype=torch.long),
            "labels": torch.tensor(int(ex["label"]), dtype=torch.long),
            "weights": torch.tensor(
                float(ex.get("complexity", complexity(ex))), dtype=torch.float
            ),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_ds, dev_ds = SPRTorch(spr["train"]), SPRTorch(spr["dev"])
test_ds = SPRTorch(spr["test"])


# ---------------- model -----------------------
class RelPosBias(nn.Module):  # simple T5-style bias
    def __init__(self, heads, max_dist=128):
        super().__init__()
        self.rel = nn.Embedding(2 * max_dist, heads)
        self.max_dist = max_dist

    def forward__(self, qlen, klen):
        ctx = torch.arange(klen)[None] - torch.arange(qlen)[:, None]
        ctx = ctx.clamp(-self.max_dist, self.max_dist) + self.max_dist
        return self.rel(ctx)  # [qlen,klen,heads]

    def forward(self, qlen, klen):
        return self.forward__(qlen, klen).permute(2, 0, 1)  # [heads,qlen,klen]


class CharTransformer(nn.Module):
    def __init__(self, v, d_model=128, nhead=8, layers=4, num_cls=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(v, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        encs = []
        for _ in range(layers):
            encs.append(
                nn.TransformerEncoderLayer(d_model, nhead, ff, 0.1, batch_first=True)
            )
        self.enc = nn.ModuleList(encs)
        self.rpb = RelPosBias(nhead, max_dist=MAX_LEN)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_cls)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pos[:, : x.size(1)]
        for layer in self.enc:
            h = layer(h, src_key_padding_mask=mask)
        h = self.norm(h)
        return self.fc(h[:, 0])  # CLS token


# ---------------- utils -----------------------
def cwa(pred, lab, w):
    correct = (pred == lab).astype(float)
    return (correct * w).sum() / w.sum()


# curriculum weight schedule
def curriculum(epoch, total):
    return min(1.0, (epoch + 1) / (total / 2))  # linearly to 1 by half epochs


# ---------------- training loop ---------------
batch = 32
epochs = 12
model = CharTransformer(vsize).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "weights": [],
    }
}

best_val, bad_epochs = 1e9, 0
for epoch in range(epochs):
    model.train()
    tot_loss, items = 0, 0
    cur_w = curriculum(epoch, epochs)
    for batch_d in train_loader:
        batch_d = {k: v.to(device) for k, v in batch_d.items()}
        opt.zero_grad()
        logits = model(batch_d["input_ids"])
        loss = criterion(logits, batch_d["labels"])
        # down-weight complex examples early
        loss = (loss * torch.where(batch_d["weights"] > 5, cur_w, 1.0)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += loss.item() * batch_d["labels"].size(0)
        items += batch_d["labels"].size(0)
    train_loss = tot_loss / items
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- validation ----
    model.eval()
    vloss, vitems = 0, 0
    preds, labels, weights = [], [], []
    with torch.no_grad():
        for batch_d in dev_loader:
            batch_d = {k: v.to(device) for k, v in batch_d.items()}
            out = model(batch_d["input_ids"])
            loss = criterion(out, batch_d["labels"])
            vloss += loss.item() * batch_d["labels"].size(0)
            vitems += batch_d["labels"].size(0)
            p = out.argmax(1).cpu().numpy()
            l = batch_d["labels"].cpu().numpy()
            w = batch_d["weights"].cpu().numpy()
            preds.extend(p)
            labels.extend(l)
            weights.extend(w)
    vloss /= vitems
    mf1 = f1_score(labels, preds, average="macro")
    cw = cwa(np.array(preds), np.array(labels), np.array(weights))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vloss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append({"macro_f1": mf1, "cwa": cw})
    print(
        f"Epoch {epoch+1}: validation_loss = {vloss:.4f} | Macro-F1={mf1:.3f} | CWA={cw:.3f}"
    )
    # early stopping
    if vloss < best_val - 1e-4:
        best_val = vloss
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs >= 3:
        print("Early stopping.")
        break
    sched.step()

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels
experiment_data["SPR_BENCH"]["weights"] = weights
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
