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

import os, time, random, pathlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from collections import Counter

# ------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# metrics -----------------------------------------------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(w0 for w0, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(w0 for w0, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(w0 for w0, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ------------------------------------------------------------------
# SPR-BENCH loader --------------------------------------------------
def resolve_spr_path() -> pathlib.Path:
    for p in [
        os.getenv("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.cwd().parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if p and (pathlib.Path(p) / "train.csv").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("Could not locate SPR_BENCH")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr_path = resolve_spr_path()
spr = load_spr_bench(spr_path)
print({k: len(v) for k, v in spr.items()})


# ------------------------------------------------------------------
# vocabulary --------------------------------------------------------
def tokenize(s):
    return s.strip().split()


vocab_counter = Counter(tok for s in spr["train"]["sequence"] for tok in tokenize(s))
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = 0, 1


def encode_tokens(toks):
    return [stoi.get(t, unk_idx) for t in toks]


def encode_seq(seq):
    return encode_tokens(tokenize(seq))


labels = sorted(set(spr["train"]["label"]))
ltoi = {l: i for i, l in enumerate(labels)}
itos_l = {i: l for l, i in ltoi.items()}

# ------------------------------------------------------------------
# dataset wrappers --------------------------------------------------
MAX_LEN = 128


class SupervisedSPR(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labs = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(
                encode_seq(self.seqs[idx])[:MAX_LEN], dtype=torch.long
            ),
            "label": torch.tensor(self.labs[idx], dtype=torch.long),
        }


def collate_supervised(batch):
    mx = max(len(b["input"]) for b in batch)
    x = torch.full((len(batch), mx), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["input"])] = b["input"]
    y = torch.stack([b["label"] for b in batch])
    return {"input": x.to(device), "label": y.to(device)}


# ------------------------------------------------------------------
# transformer encoder ----------------------------------------------
class SPRTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_heads=4, n_layers=2, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.emb_dim = emb_dim

    def forward(self, x):
        pos_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos_ids)
        mask = x == pad_idx
        h = self.encoder(h, src_key_padding_mask=mask)
        m = (~mask).unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.enc = encoder
        self.clf = nn.Linear(encoder.emb_dim, num_labels)

    def forward(self, x):
        return self.clf(self.enc(x))


# ------------------------------------------------------------------
# experiment data dict ---------------------------------------------
experiment_data = {
    "supervised_only": {
        "SPR_BENCH": {
            "metrics": {"val_SWA": [], "val_CWA": [], "val_SCWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ------------------------------------------------------------------
# training loop -----------------------------------------------------
emb_dim = 128
model = SPRModel(
    SPRTransformer(len(vocab), emb_dim=emb_dim).to(device), len(labels)
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    SupervisedSPR(spr["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_supervised,
)
val_loader = DataLoader(
    SupervisedSPR(spr["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_supervised,
)

best_scwa, best_preds, best_trues = -1, [], []
epochs = 6
for ep in range(1, epochs + 1):
    # ---- train ----
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        opt.zero_grad()
        loss = criterion(model(batch["input"]), batch["label"])
        loss.backward()
        opt.step()
        tr_loss += loss.item() * batch["label"].size(0)
    tr_loss /= len(train_loader.dataset)
    # ---- val ----
    model.eval()
    val_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input"])
            val_loss += criterion(logits, batch["label"]).item() * batch["label"].size(
                0
            )
            preds += logits.argmax(1).cpu().tolist()
            trues += batch["label"].cpu().tolist()
    val_loss /= len(val_loader.dataset)
    swa = shape_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    cwa = color_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
    scwa_v = scwa(spr["dev"]["sequence"], trues, preds)
    # record
    ed = experiment_data["supervised_only"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["val_SWA"].append(swa)
    ed["metrics"]["val_CWA"].append(cwa)
    ed["metrics"]["val_SCWA"].append(scwa_v)
    ed["timestamps"].append(time.time())
    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} SCWA={scwa_v:.4f}"
    )
    if scwa_v > best_scwa:
        best_scwa, best_preds, best_trues = scwa_v, preds, trues

# store best preds / trues
ed["predictions"] = best_preds
ed["ground_truth"] = best_trues

# ------------------------------------------------------------------
# save experiment data ---------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Ablation experiment data saved to working/experiment_data.npy")
