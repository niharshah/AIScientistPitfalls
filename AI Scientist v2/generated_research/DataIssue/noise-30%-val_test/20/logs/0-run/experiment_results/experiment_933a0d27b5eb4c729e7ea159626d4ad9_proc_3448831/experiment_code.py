import os, pathlib, math, random, string, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- paths / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# ---------- data ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH missing -> synthesising toy data")

    def synth(n):
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

# ---------- vocab / encoding ----------
vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vsize = len(vocab)
MAX_LEN = min(max(len(ex["sequence"]) for ex in spr["train"]) + 1, 128)


def enc(seq):
    ids = [vocab["<cls>"]] + [vocab.get(c, 1) for c in seq][: MAX_LEN - 1]
    return ids + [0] * (MAX_LEN - len(ids))


def complexity(ex):  # proxy
    return float(len(set(ex["sequence"])))


class SPRTorch(Dataset):
    def __init__(self, d):
        self.d = d

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        ex = self.d[i]
        return {
            "input_ids": torch.tensor(enc(ex["sequence"]), dtype=torch.long),
            "labels": torch.tensor(int(ex["label"]), dtype=torch.long),
            "weights": torch.tensor(
                float(ex.get("complexity", complexity(ex))), dtype=torch.float
            ),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_ds, dev_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)


# ---------- model ----------
class RelPosBias(nn.Module):
    def __init__(self, heads, max_dist=128):
        super().__init__()
        self.rel = nn.Embedding(2 * max_dist, heads)
        self.m = max_dist

    def forward(self, q, k):
        ctx = torch.arange(k)[None] - torch.arange(q)[:, None]
        ctx = ctx.clamp(-self.m, self.m) + self.m
        return self.rel(ctx).permute(2, 0, 1)


class CharTransformer(nn.Module):
    def __init__(self, v, d=128, h=8, layers=4, ff=256):
        super().__init__()
        self.emb = nn.Embedding(v, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d))
        self.enc = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d, h, ff, 0.1, batch_first=True)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 2)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pos[:, : x.size(1)]
        for layer in self.enc:
            h = layer(h, src_key_padding_mask=mask)
        return self.fc(self.norm(h)[:, 0])


def cwa(pred, lab, w):
    return ((pred == lab).astype(float) * w).sum() / w.sum()


def curriculum(ep, total):
    return min(1.0, (ep + 1) / (total / 2))


# ---------- training ----------
batch, epochs = 32, 12
model = CharTransformer(vsize).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

experiment_data = {
    "no_grad_clip": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "weights": [],
        }
    }
}

best_val, bad_epochs = 1e9, 0
for ep in range(epochs):
    model.train()
    tot_loss = items = 0
    cur_w = curriculum(ep, epochs)
    for bd in train_loader:
        bd = {k: v.to(device) for k, v in bd.items()}
        opt.zero_grad()
        logits = model(bd["input_ids"])
        loss = criterion(logits, bd["labels"])
        loss = (loss * torch.where(bd["weights"] > 5, cur_w, 1.0)).mean()
        loss.backward()
        # ---- NO GRADIENT CLIPPING HERE ----
        opt.step()
        tot_loss += loss.item() * bd["labels"].size(0)
        items += bd["labels"].size(0)
    tr_loss = tot_loss / items
    experiment_data["no_grad_clip"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # validation
    model.eval()
    vloss = vitems = 0
    preds = []
    labs = []
    ws = []
    with torch.no_grad():
        for bd in dev_loader:
            bd = {k: v.to(device) for k, v in bd.items()}
            out = model(bd["input_ids"])
            loss = criterion(out, bd["labels"])
            vloss += loss.item() * bd["labels"].size(0)
            vitems += bd["labels"].size(0)
            p = out.argmax(1).cpu().numpy()
            l = bd["labels"].cpu().numpy()
            w = bd["weights"].cpu().numpy()
            preds.extend(p)
            labs.extend(l)
            ws.extend(w)
    vloss /= vitems
    mf1 = f1_score(labs, preds, average="macro")
    cw = cwa(np.array(preds), np.array(labs), np.array(ws))
    experiment_data["no_grad_clip"]["SPR_BENCH"]["losses"]["val"].append(vloss)
    experiment_data["no_grad_clip"]["SPR_BENCH"]["metrics"]["val"].append(
        {"macro_f1": mf1, "cwa": cw}
    )
    print(f"Ep {ep+1}: val_loss={vloss:.4f} Macro-F1={mf1:.3f} CWA={cw:.3f}")
    if vloss < best_val - 1e-4:
        best_val = vloss
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs >= 3:
        print("Early stopping.")
        break
    sched.step()

experiment_data["no_grad_clip"]["SPR_BENCH"]["predictions"] = np.array(preds)
experiment_data["no_grad_clip"]["SPR_BENCH"]["ground_truth"] = np.array(labs)
experiment_data["no_grad_clip"]["SPR_BENCH"]["weights"] = np.array(ws)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved to", os.path.join(working_dir, "experiment_data.npy"))
