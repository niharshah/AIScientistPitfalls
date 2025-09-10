import os, pathlib, random, math, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# basic folders & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# (BUG-FIX) make sure we ALWAYS have the benchmark folder -------------
def _make_synthetic_csv(path: pathlib.Path, rows: int):
    shapes = "ABCD"  # 4 shapes
    colors = "abcd"  # 4 colours
    rules = ["X", "Y"]  # two dummy labels
    with path.open("w") as f:
        f.write("id,sequence,label\n")
        for i in range(rows):
            seq_len = random.randint(3, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(seq_len)
            )
            label = random.choice(rules)
            f.write(f"{i},{seq},{label}\n")


def locate_or_create_spr() -> pathlib.Path:
    """Return a path to SPR_BENCH, creating a synthetic one if necessary."""
    search_roots = [os.getenv("SPR_DIR"), ".", "..", "../..", working_dir]
    for root in search_roots:
        if root and (pathlib.Path(root) / "SPR_BENCH" / "train.csv").exists():
            return pathlib.Path(root) / "SPR_BENCH"
    # not found: create tiny synthetic dataset
    syn_root = pathlib.Path(working_dir) / "SPR_BENCH"
    syn_root.mkdir(parents=True, exist_ok=True)
    _make_synthetic_csv(syn_root / "train.csv", 400)
    _make_synthetic_csv(syn_root / "dev.csv", 100)
    _make_synthetic_csv(syn_root / "test.csv", 200)
    print(f"Created synthetic SPR_BENCH at {syn_root.resolve()}")
    return syn_root


# ---------------------------------------------------------------------
# data loading & metrics ----------------------------------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _ld(split):
        return load_dataset(
            "csv",
            data_files=str(root / f"{split}.csv"),
            split="train",
            cache_dir=os.path.join(working_dir, ".cache_dsets"),
        )

    return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))


def count_shape(seq: str):
    return len({tok[0] for tok in seq.split() if tok})


def count_color(seq: str):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def swa(seqs, y_true, y_pred):
    wts = [count_shape(s) for s in seqs]
    cor = [w if t == p else 0 for w, t, p in zip(wts, y_true, y_pred)]
    return sum(cor) / sum(wts) if sum(wts) else 0.0


# ---------------------------------------------------------------------
# dataset class --------------------------------------------------------
class SPRSet(Dataset):
    def __init__(self, split, tok2id, lbl2id, max_len=40):
        self.data, self.tok2id, self.lbl2id, self.max_len = (
            split,
            tok2id,
            lbl2id,
            max_len,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        toks = row["sequence"].split()
        ids = [self.tok2id.get(t, self.tok2id["<unk>"]) for t in toks][: self.max_len]
        ids += [self.tok2id["<pad>"]] * (self.max_len - len(ids))
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(min(len(toks), self.max_len), dtype=torch.long),
            "shape_c": torch.tensor(count_shape(row["sequence"]), dtype=torch.float),
            "color_c": torch.tensor(count_color(row["sequence"]), dtype=torch.float),
            "label": torch.tensor(self.lbl2id[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


# ---------------------------------------------------------------------
# model ----------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridSPR(nn.Module):
    def __init__(self, vocab, n_cls, d_model=128, n_head=4, n_layers=2, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=d_model * 2, dropout=0.1, batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.sym = nn.Linear(3, d_model // 2)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model + d_model // 2),
            nn.Linear(d_model + d_model // 2, n_cls),
        )

    def forward(self, ids, lens, shape_c, color_c):
        x = self.emb(ids) * math.sqrt(self.emb.embedding_dim)
        x = self.pos(x)
        kp_mask = ids.eq(0)
        x = self.tr(x, src_key_padding_mask=kp_mask)
        m = (~kp_mask).unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-6)
        sym = torch.stack([shape_c, color_c, lens.float()], 1)
        sym = self.sym(sym)
        return self.out(torch.cat([pooled, sym], 1))


# ---------------------------------------------------------------------
# preparations ---------------------------------------------------------
root_path = locate_or_create_spr()
spr = load_spr(root_path)

spec_tokens = ["<pad>", "<unk>"]
vocab = sorted({tok for s in spr["train"]["sequence"] for tok in s.split()})
tok2id = {t: i + len(spec_tokens) for i, t in enumerate(vocab)}
tok2id["<pad>"] = 0
tok2id["<unk>"] = 1
lbls = sorted(set(spr["train"]["label"]))
lbl2id = {l: i for i, l in enumerate(lbls)}
id2lbl = {i: l for l, i in lbl2id.items()}

train_ds = SPRSet(spr["train"], tok2id, lbl2id)
dev_ds = SPRSet(spr["dev"], tok2id, lbl2id)
test_ds = SPRSet(spr["test"], tok2id, lbl2id)
train_dl = DataLoader(train_ds, 128, shuffle=True)
dev_dl = DataLoader(dev_ds, 256)
test_dl = DataLoader(test_ds, 256)


# ---------------------------------------------------------------------
# training loop --------------------------------------------------------
def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, tot = 0.0, 0
    ytrue, ypred, seqs = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["ids"], batch["len"], batch["shape_c"], batch["color_c"])
        loss = crit(out, batch["label"])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        bs = batch["label"].size(0)
        tot_loss += loss.item() * bs
        tot += bs
        ytrue += batch["label"].cpu().tolist()
        ypred += out.argmax(1).cpu().tolist()
        seqs += batch["raw_seq"]
    y_true_lbl = [id2lbl[i] for i in ytrue]
    y_pred_lbl = [id2lbl[i] for i in ypred]
    return tot_loss / tot, swa(seqs, y_true_lbl, y_pred_lbl)


def train_model(layers, heads, epochs=5):
    model = HybridSPR(len(tok2id), len(lbls), 128, heads, layers, pad_idx=0).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_swa, best_state = -1.0, None
    hist = {"train_loss": [], "train_swa": [], "val_loss": [], "val_swa": []}
    for ep in range(1, epochs + 1):
        tr_loss, tr_swa = run_epoch(model, train_dl, crit, opt)
        vl_loss, vl_swa = run_epoch(model, dev_dl, crit)
        hist["train_loss"].append(tr_loss)
        hist["train_swa"].append(tr_swa)
        hist["val_loss"].append(vl_loss)
        hist["val_swa"].append(vl_swa)
        print(f"Epoch {ep}: validation_loss = {vl_loss:.4f}, SWA={vl_swa:.4f}")
        if vl_swa > best_swa:
            best_swa, best_state = vl_swa, {
                k: v.cpu() for k, v in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    test_loss, test_swa = run_epoch(model, test_dl, crit)
    return model, hist, test_loss, test_swa


# ---------------------------------------------------------------------
# experiment -----------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": [], "test_swa": None},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
    }
}

model, hist, t_loss, t_swa = train_model(layers=2, heads=4, epochs=5)
experiment_data["SPR_BENCH"]["metrics"]["train_swa"] = hist["train_swa"]
experiment_data["SPR_BENCH"]["metrics"]["val_swa"] = hist["val_swa"]
experiment_data["SPR_BENCH"]["metrics"]["test_swa"] = t_swa
experiment_data["SPR_BENCH"]["losses"]["train"] = hist["train_loss"]
experiment_data["SPR_BENCH"]["losses"]["val"] = hist["val_loss"]
experiment_data["SPR_BENCH"]["losses"]["test"] = t_loss

# make predictions on test set for saving
model.eval()
with torch.no_grad():
    for batch in test_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["ids"], batch["len"], batch["shape_c"], batch["color_c"])
        experiment_data["SPR_BENCH"]["ground_truth"] += [
            id2lbl[i] for i in batch["label"].cpu().tolist()
        ]
        experiment_data["SPR_BENCH"]["predictions"] += [
            id2lbl[i] for i in out.argmax(1).cpu().tolist()
        ]

print(f"Test SWA = {t_swa:.4f}")
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
