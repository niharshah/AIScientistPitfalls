import os, json, datetime, random, string
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- working dir & experiment dict ------------------
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

# ------------------------------ device -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------ load or create data ------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def has_real_data(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )


if has_real_data(SPR_PATH):
    print("Loading real SPR_BENCH …")
    from datasets import load_dataset, DatasetDict

    def load_spr(root):
        ds = DatasetDict()
        for sp in ["train", "dev", "test"]:
            ds[sp] = load_dataset(
                "csv",
                data_files=os.path.join(root, f"{sp}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )
        return ds

    dsets = load_spr(SPR_PATH)
    raw = {
        sp: {"sequence": dsets[sp]["sequence"], "label": dsets[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }
else:
    print("No dataset found – generating small synthetic corpus.")
    shapes = list(string.ascii_uppercase[:6])
    colours = list("0123")

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colours)
            for _ in range(random.randint(4, 10))
        )

    def rule(seq):
        us = len({tok[0] for tok in seq.split()})
        uc = len({tok[1] for tok in seq.split()})
        return int(us >= uc)  # arbitrary unseen rule

    def make(n):
        xs = [rand_seq() for _ in range(n)]
        return {"sequence": xs, "label": [rule(x) for x in xs]}

    raw = {"train": make(3000), "dev": make(600), "test": make(800)}


# ------------------- symbolic feature helpers -----------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def shape_weighted_accuracy(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == pt else 0 for wi, yt, pt in zip(w, y, p)) / (sum(w) or 1)


PAD, UNK = "<PAD>", "<UNK>"
shape_set = sorted({tok[0] for seq in raw["train"]["sequence"] for tok in seq.split()})
colour_set = sorted({tok[1] for seq in raw["train"]["sequence"] for tok in seq.split()})
shape2idx = {s: i for i, s in enumerate(shape_set)}
colour2idx = {c: i for i, c in enumerate(colour_set)}


def sym_vec(seq: str):
    shp = [0] * len(shape_set)
    col = [0] * len(colour_set)
    for tok in seq.split():
        if tok[0] in shape2idx:
            shp[shape2idx[tok[0]]] += 1
        if tok[1] in colour2idx:
            col[colour2idx[tok[1]]] += 1
    nus, nuc = sum(1 for x in shp if x), sum(1 for x in col if x)
    eq = int(nus == nuc)
    return shp + col + [nus, nuc, eq]


SYM_DIM = len(shape_set) + len(colour_set) + 3


# ----------------------- vocab & encoding ----------------------------
def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    for tok in {t for s in seqs for t in s.split()}:
        vocab.setdefault(tok, len(vocab))
    return vocab


vocab = build_vocab(raw["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, 1) for tok in seq.split()]


# --------------------------- Dataset ---------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.raw = seqs
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.S = [torch.tensor(sym_vec(s), dtype=torch.float32) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"ids": self.X[idx], "sym": self.S[idx], "label": self.y[idx]}


def collate(batch):
    maxlen = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    mask = torch.ones((len(batch), maxlen), dtype=torch.bool)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
        mask[i, : len(b["ids"])] = False
    sym = torch.stack([b["sym"] for b in batch])
    lab = torch.stack([b["label"] for b in batch])
    lens = torch.tensor([len(b["ids"]) for b in batch])
    return {"ids": ids, "mask": mask, "sym": sym, "labels": lab, "lens": lens}


datasets = {
    sp: SPRDataset(raw[sp]["sequence"], raw[sp]["label"])
    for sp in ["train", "dev", "test"]
}
loaders = {
    sp: DataLoader(
        datasets[sp], batch_size=64, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in ["train", "dev", "test"]
}


# ----------------------------- Model ---------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe[None])

    def forward(self, x):  # x: B,L,E
        return x + self.pe[:, : x.size(1)]


class NeuroSymTransformer(nn.Module):
    def __init__(
        self, vocab_sz, emb=64, nhead=4, nlayers=2, tf_hid=128, sym_hid=64, n_cls=2
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=vocab[PAD])
        self.pos = PositionalEncoding(emb)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=nhead, dim_feedforward=tf_hid, batch_first=True
        )
        self.trf = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.symb = nn.Sequential(nn.Linear(SYM_DIM, sym_hid), nn.ReLU())
        self.cls = nn.Linear(emb + sym_hid, n_cls)

    def forward(self, ids, mask, sym):
        x = self.emb(ids)
        x = self.pos(x)
        h = self.trf(x, src_key_padding_mask=mask)  # [B,L,E]
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(
            min=1
        )  # mean pool
        s = self.symb(sym)
        out = torch.cat([pooled, s], dim=1)
        return self.cls(out)


model = NeuroSymTransformer(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# ----------------------- evaluation util -----------------------------
@torch.no_grad()
def do_eval(split):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts = [], []
    for batch in loaders[split]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch["ids"], batch["mask"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss_sum += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1)
        preds.extend(p.cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
        correct += (p == batch["labels"]).sum().item()
        tot += batch["labels"].size(0)
    acc = correct / tot
    swa = shape_weighted_accuracy(datasets[split].raw, gts, preds)
    return acc, loss_sum / tot, swa, preds, gts


# --------------------------- training loop ---------------------------
best_val, patience, wait = float("inf"), 2, 0
best_state = None
for epoch in range(1, 16):
    model.train()
    run_loss, seen = 0.0, 0
    for batch in loaders["train"]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["ids"], batch["mask"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)
    tr_acc, _, tr_swa, _, _ = do_eval("train")
    val_acc, val_loss, val_swa, _, _ = do_eval("dev")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(run_loss / seen)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"acc": tr_acc, "swa": tr_swa}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "swa": val_swa}
    )
    experiment_data["SPR_BENCH"]["timestamps"].append(str(datetime.datetime.now()))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.3f}")
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ------------------------------ test ---------------------------------
if best_state:
    model.load_state_dict(best_state)
test_acc, test_loss, test_swa, preds, gts = do_eval("test")
print(f"TEST: Acc={test_acc:.3f} | SWA={test_swa:.3f}")
experiment_data["SPR_BENCH"]["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# ---------------------------- persist --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as f:
    json.dump(experiment_data, f, indent=2)
