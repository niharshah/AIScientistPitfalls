import os, random, string, json, datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------------------- device ----------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- data loading --------------------------- #
def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ("train", "dev", "test")
    )


SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")

if spr_files_exist(SPR_PATH):
    from datasets import load_dataset, DatasetDict

    def load_spr(root):
        d = DatasetDict()
        for sp in ("train", "dev", "test"):
            d[sp] = load_dataset(
                "csv", data_files=os.path.join(root, f"{sp}.csv"), split="train"
            )
        return d

    ds = load_spr(SPR_PATH)
    raw = {
        sp: {"sequence": ds[sp]["sequence"], "label": ds[sp]["label"]}
        for sp in ("train", "dev", "test")
    }
else:  # fall back to synthetic toy set
    print("Real dataset not found – using synthetic toy data.")
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colours = [str(i) for i in range(4)]  # 0-3

    def rand_seq():
        ln = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colours) for _ in range(ln)
        )

    def rule(seq):
        us = len(set(t[0] for t in seq.split()))
        uc = len(set(t[1] for t in seq.split()))
        return int(us == uc)

    def make(n):
        xs = [rand_seq() for _ in range(n)]
        ys = [rule(s) for s in xs]
        return {"sequence": xs, "label": ys}

    raw = {"train": make(3000), "dev": make(600), "test": make(800)}

# Optionally subsample train for speed (keep ≤5000)
max_train = 5000
if len(raw["train"]["sequence"]) > max_train:
    idx = np.random.choice(len(raw["train"]["sequence"]), max_train, replace=False)
    raw["train"]["sequence"] = [raw["train"]["sequence"][i] for i in idx]
    raw["train"]["label"] = [raw["train"]["label"][i] for i in idx]


# --------------------- symbolic feature helpers --------------------- #
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred))
    return num / (sum(w) or 1)


PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    tokens = {tok for s in seqs for tok in s.split()}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(tokens))})
    return vocab


vocab = build_vocab(raw["train"]["sequence"])

shape_set = sorted({tok[0] for s in raw["train"]["sequence"] for tok in s.split()})
colour_set = sorted({tok[1] for s in raw["train"]["sequence"] for tok in s.split()})
shape2idx = {s: i for i, s in enumerate(shape_set)}
colour2idx = {c: i for i, c in enumerate(colour_set)}
SYM_DIM = len(shape_set) + len(colour_set) + 3


def sym_features(seq: str):
    shp = [0] * len(shape_set)
    col = [0] * len(colour_set)
    for tok in seq.split():
        if tok[0] in shape2idx:
            shp[shape2idx[tok[0]]] += 1
        if tok[1] in colour2idx:
            col[colour2idx[tok[1]]] += 1
    n_us = sum(x > 0 for x in shp)
    n_uc = sum(x > 0 for x in col)
    eq = 1 if n_us == n_uc else 0
    return shp + col + [n_us, n_uc, eq]


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ------------------------- Torch Dataset ---------------------------- #
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.raw_seq = seqs
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.S = [torch.tensor(sym_features(s), dtype=torch.float32) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"ids": self.X[idx], "sym": self.S[idx], "label": self.y[idx]}


def collate(batch):
    maxlen = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
    lens = torch.tensor([len(b["ids"]) for b in batch])
    sym = torch.stack([b["sym"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"ids": ids, "lens": lens, "sym": sym, "labels": labels}


datasets = {
    sp: SPRDataset(raw[sp]["sequence"], raw[sp]["label"])
    for sp in ("train", "dev", "test")
}
loaders = {
    sp: DataLoader(
        datasets[sp], batch_size=64, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in ("train", "dev", "test")
}


# --------------------------- Model ---------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FiLMTransformer(nn.Module):
    def __init__(
        self, vocab_sz, emb_dim, n_heads, n_layers, sym_dim, hid_sym, n_classes
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=vocab[PAD])
        self.pos = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.mod = nn.Sequential(
            nn.Linear(sym_dim, hid_sym), nn.ReLU(), nn.Linear(hid_sym, emb_dim * 2)
        )
        self.cls = nn.Linear(emb_dim, n_classes)

    def forward(self, ids, lens, sym):
        mask = ids == vocab[PAD]
        x = self.emb(ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        pooled = x.sum(1) / lens.unsqueeze(1)  # mean pool
        gamma_beta = self.mod(sym)  # [B,2*emb]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        modulated = gamma * pooled + beta
        return self.cls(modulated)


model = FiLMTransformer(
    len(vocab), 64, n_heads=4, n_layers=2, sym_dim=SYM_DIM, hid_sym=64, n_classes=2
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------ evaluation -------------------------------- #
@torch.no_grad()
def evaluate(split):
    model.eval()
    total, loss_sum = 0, 0.0
    preds, gts = [], []
    for batch in loaders[split]:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["ids"], batch["lens"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss_sum += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1)
        preds.extend(p.cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
        total += batch["labels"].size(0)
    swa = shape_weighted_accuracy(datasets[split].raw_seq, gts, preds)
    acc = (np.array(gts) == np.array(preds)).mean()
    return acc, loss_sum / total, swa, preds, gts


# ------------------------ training loop ----------------------------- #
best_val_loss = float("inf")
patience = 3
wait = 0
best_state = None
for epoch in range(1, 16):
    model.train()
    epoch_loss = 0
    seen = 0
    for batch in loaders["train"]:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["ids"], batch["lens"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)
    train_loss = epoch_loss / seen
    tr_acc, _, tr_swa, _, _ = evaluate("train")
    val_acc, val_loss, val_swa, _, _ = evaluate("dev")

    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": tr_acc, "swa": tr_swa})
    ed["metrics"]["val"].append({"acc": val_acc, "swa": val_swa})
    ed["timestamps"].append(str(datetime.datetime.now()))

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.3f}")
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------------------- final evaluation ---------------------------- #
if best_state:
    model.load_state_dict(best_state)
test_acc, test_loss, test_swa, preds, gts = evaluate("test")
print(f"TEST: Acc={test_acc:.3f} | SWA={test_swa:.3f}")

experiment_data["SPR_BENCH"]["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)
