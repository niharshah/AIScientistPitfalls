import os, math, pathlib, numpy as np, torch, random
from collections import Counter
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ----------------------- working dir & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------- experiment container -------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_swa": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
        "timestamps": [],
    }
}

# ----------------------- helper functions -----------------------
PAD, UNK = "<pad>", "<unk>"


def resolve_spr_path() -> pathlib.Path:
    env = os.getenv("SPR_PATH")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        return fallback
    raise FileNotFoundError("Cannot locate SPR_BENCH dataset")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _load(name):  # load each csv as a single split called 'train'
        return load_dataset(
            "csv",
            data_files=str(root / name),
            split="train",
            cache_dir=str(working_dir) + "/.hf_cache",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    tot_w = sum(weights)
    return sum(correct) / (tot_w if tot_w else 1.0)


# ----------------------- dataset & vocab ------------------------
DATA_PATH = resolve_spr_path()
dsets = load_spr(DATA_PATH)

v_counter = Counter(tok for seq in dsets["train"]["sequence"] for tok in seq.split())
vocab = {PAD: 0, UNK: 1}
for tok in v_counter:
    vocab.setdefault(tok, len(vocab))
id2tok = {i: t for t, i in vocab.items()}

label_set = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(label_set)}
num_classes = len(label_set)
print(f"Vocab size={len(vocab)} | #classes={num_classes}")


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode(s), dtype=torch.long),
            "sym_feats": torch.tensor(
                [len(s.split()), count_shape_variety(s), count_color_variety(s)],
                dtype=torch.float32,
            ),
            "shape_var": torch.tensor(count_shape_variety(s), dtype=torch.float32),
            "color_var": torch.tensor(count_color_variety(s), dtype=torch.float32),
            "label": torch.tensor(lab2id[self.labels[idx]], dtype=torch.long),
            "seq_str": s,
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    sym_feats = torch.stack([b["sym_feats"] for b in batch])
    shape_var = torch.stack([b["shape_var"] for b in batch])
    color_var = torch.stack([b["color_var"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {
        "input_ids": input_ids,
        "sym_feats": sym_feats,
        "shape_var": shape_var,
        "color_var": color_var,
        "labels": labels,
        "seqs": seqs,
    }


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

# ----------------------- model ----------------------------------
EMB_DIM = 128
SYM_DIM = 3
SYM_PROJ = 32
N_HEAD = 4
N_LAY = 2
DROP = 0.2
AUX_LAMBDA = 0.1


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NeuroSymbolic(nn.Module):
    def __init__(self, vocab_sz, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, EMB_DIM, padding_idx=0)
        self.pos = PosEnc(EMB_DIM)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMB_DIM,
            nhead=N_HEAD,
            dim_feedforward=EMB_DIM * 2,
            dropout=DROP,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=N_LAY)
        self.sym_proj = nn.Sequential(nn.Linear(SYM_DIM, SYM_PROJ), nn.ReLU())
        self.dropout = nn.Dropout(DROP)
        hidden_dim = EMB_DIM + SYM_PROJ
        self.cls_head = nn.Linear(hidden_dim, num_labels)
        self.shape_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, 1)

    def forward(self, ids, sym):
        mask = ids == 0
        x = self.emb(ids)
        x = self.pos(x)
        h = self.enc(x, src_key_padding_mask=mask)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1).clamp(
            min=1e-6
        ).unsqueeze(-1)
        z = torch.cat([pooled, self.sym_proj(sym)], dim=-1)
        z = self.dropout(z)
        return (
            self.cls_head(z),
            self.shape_head(z).squeeze(-1),
            self.color_head(z).squeeze(-1),
        )


model = NeuroSymbolic(len(vocab), num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()


# ----------------------- train / eval ---------------------------
def evaluate(loader):
    model.eval()
    total, n = 0.0, 0
    all_pred, all_lab, all_seq = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits, pred_shape, pred_color = model(
                batch["input_ids"], batch["sym_feats"]
            )
            loss = ce_loss(logits, batch["labels"]) + AUX_LAMBDA * (
                mse_loss(pred_shape, batch["shape_var"])
                + mse_loss(pred_color, batch["color_var"])
            )
            bs = batch["labels"].size(0)
            total += loss.item() * bs
            n += bs
            preds = logits.argmax(1).cpu().tolist()
            labs = batch["labels"].cpu().tolist()
            seqs = batch["seqs"]
            all_pred.extend(preds)
            all_lab.extend(labs)
            all_seq.extend(seqs)
    swa = shape_weighted_accuracy(all_seq, all_lab, all_pred)
    return total / n, swa, all_pred, all_lab


EPOCHS = 25
PATIENCE = 4
best_swa, patience = -1, 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_tot, seen = 0.0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits, p_shape, p_color = model(batch["input_ids"], batch["sym_feats"])
        loss = ce_loss(logits, batch["labels"]) + AUX_LAMBDA * (
            mse_loss(p_shape, batch["shape_var"])
            + mse_loss(p_color, batch["color_var"])
        )
        loss.backward()
        optimizer.step()
        bs = batch["labels"].size(0)
        tr_tot += loss.item() * bs
        seen += bs
    val_loss, val_swa, _, _ = evaluate(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.4f}")
    # log
    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append(tr_tot / seen)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(val_swa)
    experiment_data["SPR_BENCH"]["timestamps"].append(datetime.utcnow().isoformat())
    # early stop
    if val_swa > best_swa:
        best_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1
    if patience >= PATIENCE:
        print("Early stopping.")
        break

# ----------------------- final evaluation -----------------------
if best_state:
    model.load_state_dict(best_state)
dev_loss, dev_swa, dev_pred, dev_lab = evaluate(dev_loader)
test_loss, test_swa, test_pred, test_lab = evaluate(test_loader)

print(f"=== DEV  === loss {dev_loss:.4f} | SWA {dev_swa:.4f}")
print(f"=== TEST === loss {test_loss:.4f} | SWA {test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_pred
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_lab
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_lab

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
