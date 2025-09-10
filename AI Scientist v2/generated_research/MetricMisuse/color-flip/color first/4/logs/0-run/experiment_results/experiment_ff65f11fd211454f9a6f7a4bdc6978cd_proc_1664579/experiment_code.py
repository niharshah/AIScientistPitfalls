import os, pathlib, random, math, time, copy, warnings, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.cluster import KMeans

# --------------- mandatory working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- device handling -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- experiment store ---------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "raw_sequences": [],
    }
}


# ---------------- dataset helpers ----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(split_file):
        return load_dataset(
            "csv",
            data_files=str(root / split_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def synthetic_dataset() -> DatasetDict:
    shapes, colors = ["▲", "■", "●", "◆"], list("RGBY")

    def gen(n):
        seq, lab, ids = [], [], []
        for i in range(n):
            ids.append(str(i))
            toks = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 10))
            ]
            seq.append(" ".join(toks))
            lab.append(random.choice(["ruleA", "ruleB", "ruleC"]))
        return Dataset.from_dict({"id": ids, "sequence": seq, "label": lab})

    return DatasetDict(train=gen(800), dev=gen(160), test=gen(160))


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
spr = load_spr_bench(data_root) if data_root.exists() else synthetic_dataset()
print({k: len(v) for k, v in spr.items()})


# ------------- basic metric utils ----------------------
def count_color(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def CWA(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if w else 0.0


def SWA(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if w else 0.0


def CompWA(seqs, y_t, y_p):
    w = [count_color(s) * count_shape(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if w else 0.0


# ------------- glyph clustering ------------------------
all_tokens = set()
for s in spr["train"]["sequence"]:
    all_tokens.update(s.split())
shapes = list(sorted(set(t[0] for t in all_tokens)))
colors = list(sorted(set(t[1] for t in all_tokens)))
shape2id = {s: i for i, s in enumerate(shapes)}
color2id = {c: i for i, c in enumerate(colors)}
# build feature vector [shape_id one-hot || color_id one-hot]
feat = []
tok_list = []
for t in sorted(all_tokens):
    v = np.zeros(len(shapes) + len(colors), dtype=np.float32)
    v[shape2id[t[0]]] = 1.0
    v[len(shapes) + color2id[t[1]]] = 1.0
    feat.append(v)
    tok_list.append(t)
feat = np.stack(feat)
n_clusters = min(16, len(tok_list))  # adjustable
km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(feat)
token2cluster = {
    tok: int(cid) + 1 for tok, cid in zip(tok_list, km.labels_)
}  # 0 reserved


def remap(example):
    return {
        "sequence": " ".join(str(token2cluster[t]) for t in example["sequence"].split())
    }


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(remap, batched=False, load_from_cache_file=False)

vocab_size = n_clusters + 1
print("Latent vocab:", vocab_size - 1)


# ---------- PyTorch dataset ----------------------------
class TorchSPR(torch.utils.data.Dataset):
    def __init__(self, hf_split, label2id):
        self.ids = [s.split() for s in hf_split["sequence"]]
        self.labels = [label2id[l] for l in hf_split["label"]]
        self.raw = hf_split["sequence"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                [int(x) for x in self.ids[idx]], dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.raw[idx],
        }


def collate(batch):
    maxlen = max(len(x["input_ids"]) for x in batch)
    ids = torch.stack(
        [
            nn.functional.pad(
                x["input_ids"], (0, maxlen - len(x["input_ids"])), value=0
            )
            for x in batch
        ]
    )
    labels = torch.stack([x["labels"] for x in batch])
    raws = [x["raw"] for x in batch]
    return {"input_ids": ids, "labels": labels, "raw": raws}


label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
train_loader = DataLoader(
    TorchSPR(spr["train"], label2id), batch_size=64, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    TorchSPR(spr["dev"], label2id), batch_size=128, shuffle=False, collate_fn=collate
)


# -------------- Model ----------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # (B,L,D)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab, dim=64, layers=2, heads=4, classes=3, drop=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=drop,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(dim, classes)

    def forward(self, x):
        mask = x == 0
        h = self.pos(self.embed(x))
        h = self.enc(h, src_key_padding_mask=mask)
        h = (~mask).unsqueeze(-1) * h
        h = h.sum(1) / ((~mask).sum(1, keepdim=True) + 1e-6)
        return self.fc(h)


model = TransformerClassifier(vocab=vocab_size, classes=len(label2id)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# -------------- training loop --------------------------
epochs = 6
for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    total = 0
    tot_loss = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        loss.backward()
        optimizer.step()
        total += batch["labels"].size(0)
        tot_loss += loss.item() * batch["labels"].size(0)
    train_loss = tot_loss / total
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- validation ----
    model.eval()
    val_tot = 0
    val_loss = 0
    preds, gt, raws = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["input_ids"])
            loss = criterion(out, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            val_tot += batch["labels"].size(0)
            p = out.argmax(1).cpu().tolist()
            preds.extend(p)
            gt.extend(batch["labels"].cpu().tolist())
            raws.extend(batch["raw"])
    val_loss /= val_tot
    acc = np.mean([p == g for p, g in zip(preds, gt)])
    cwa = CWA(raws, gt, preds)
    swa = SWA(raws, gt, preds)
    comp = CompWA(raws, gt, preds)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {
            "epoch": epoch,
            "acc": float(acc),
            "CWA": float(cwa),
            "SWA": float(swa),
            "CompWA": float(comp),
        }
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
    )
    scheduler.step()

# ---------- save everything ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
