import os, math, pathlib, random, numpy as np, torch
from collections import Counter
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment container ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_swa": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
        "timestamps": [],
    }
}

# ---------- helper functions ----------
PAD, UNK = "<pad>", "<unk>"


def find_spr_path() -> pathlib.Path:
    env = os.getenv("SPR_PATH")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    default = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (default / "train.csv").exists():
        return default
    raise FileNotFoundError("SPR_BENCH not found")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def load(csv):
        return load_dataset(
            "csv",
            data_files=str(root / csv),
            split="train",
            cache_dir=str(working_dir) + "/.hf_cache",
        )

    return DatasetDict(
        train=load("train.csv"), dev=load("dev.csv"), test=load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / (sum(weights) if sum(weights) else 1.0)


# ---------- data ----------
DATA_PATH = find_spr_path()
dsets = load_spr(DATA_PATH)

vocab_counter = Counter(
    tok for seq in dsets["train"]["sequence"] for tok in seq.split()
)
vocab = {PAD: 0, UNK: 1}
for tok in vocab_counter:
    vocab.setdefault(tok, len(vocab))
id2tok = {i: t for t, i in vocab.items()}

labels = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)

print(f"Vocabulary: {len(vocab)}  |  Classes: {num_classes}")


def enc_sequence(seq: str):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labs = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        toks = torch.tensor(enc_sequence(seq), dtype=torch.long)
        sym = torch.tensor(
            [len(seq.split()), count_shape_variety(seq), count_color_variety(seq)],
            dtype=torch.float32,
        )
        return {
            "input_ids": toks,
            "sym_feats": sym,
            "label": torch.tensor(lab2id[self.labs[idx]], dtype=torch.long),
            "seq_str": seq,
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
    sym = torch.stack([b["sym_feats"] for b in batch])
    lab = torch.stack([b["label"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {"input_ids": ids, "sym_feats": sym, "labels": lab, "seqs": seqs}


BS = 128
train_loader = DataLoader(
    SPRDataset(dsets["train"]), batch_size=BS, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dsets["dev"]), batch_size=BS, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(dsets["test"]), batch_size=BS, shuffle=False, collate_fn=collate
)

# ---------- model ----------
EMB_DIM = 128
SYM_DIM = 3
SYM_PROJ = 32
TRANS_LAY = 2
HEADS = 4
AUX_W = 0.2
DROPOUT = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NeuralSymAux(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=HEADS,
            dim_feedforward=emb_dim * 2,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=TRANS_LAY)
        self.sym_proj = nn.Sequential(nn.Linear(SYM_DIM, SYM_PROJ), nn.ReLU())
        self.classifier = nn.Linear(emb_dim + SYM_PROJ, num_labels)
        self.aux_head = nn.Linear(emb_dim, 3)  # predict len, shape_var, color_var

    def forward(self, ids, sym):
        mask = ids == 0
        x = self.emb(ids)
        x = self.pos(x)
        h = self.enc(x, src_key_padding_mask=mask)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1).clamp(
            min=1e-6
        ).unsqueeze(-1)
        pooled = pooled.squeeze(-1)
        sym_emb = self.sym_proj(sym)
        logits = self.classifier(torch.cat([pooled, sym_emb], dim=-1))
        aux_pred = self.aux_head(pooled)
        return logits, aux_pred


model = NeuralSymAux(len(vocab), EMB_DIM, num_classes).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_aux = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


# ---------- training helpers ----------
def evaluate(model, loader):
    model.eval()
    total, n = 0.0, 0
    preds, labs, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits, _ = model(batch["input_ids"], batch["sym_feats"])
            loss = criterion_cls(logits, batch["labels"])
            bs = batch["labels"].size(0)
            total += loss.item() * bs
            n += bs
            p = logits.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            preds.extend(p)
            labs.extend(l)
            seqs.extend(batch["seqs"])
    swa = shape_weighted_accuracy(seqs, labs, preds)
    return total / n, swa, preds, labs


# ---------- training loop ----------
EPOCHS, PATIENCE = 20, 3
best_swa, wait, best_state = -1, 0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss, seen = 0.0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits, aux_pred = model(batch["input_ids"], batch["sym_feats"])
        cls_loss = criterion_cls(logits, batch["labels"])
        aux_tgt = batch["sym_feats"]
        aux_loss = criterion_aux(aux_pred, aux_tgt)
        loss = cls_loss + AUX_W * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = batch["labels"].size(0)
        tr_loss += loss.item() * bs
        seen += bs
    scheduler.step()
    val_loss, val_swa, _, _ = evaluate(model, dev_loader)
    print(
        f"Epoch {epoch}: train_loss={tr_loss/seen:.4f} | val_loss={val_loss:.4f} | val_SWA={val_swa:.4f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append(tr_loss / seen)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(val_swa)
    experiment_data["SPR_BENCH"]["timestamps"].append(datetime.utcnow().isoformat())
    if val_swa > best_swa:
        best_swa = val_swa
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
    if wait >= PATIENCE:
        print("Early stopping.")
        break

# ---------- final evaluation ----------
if best_state:
    model.load_state_dict(best_state)

dev_loss, dev_swa, dev_pred, dev_lab = evaluate(model, dev_loader)
test_loss, test_swa, test_pred, test_lab = evaluate(model, test_loader)
print(f"=== DEV  === loss {dev_loss:.4f} | SWA {dev_swa:.4f}")
print(f"=== TEST === loss {test_loss:.4f} | SWA {test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_pred
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_lab
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_lab

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
