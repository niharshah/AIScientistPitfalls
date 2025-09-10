# Multi-Synthetic-Dataset Training (MSDT) – single-file implementation
import os, pathlib, math, random, numpy as np, torch, time
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, concatenate_datasets

# ----------------- misc / paths / device ------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ----------------- experiment data container --------------
def _blank():
    return {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }


experiment_data = {
    "MSDT": {
        "SPR_BENCH": _blank(),
        "TOKEN_RENAMED": _blank(),
        "COLOR_SHUFFLED": _blank(),
    }
}


# ----------------- load original dataset ------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):  # each csv is a split
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr_orig = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr_orig.items()})

# ----------------- helper: lexical transforms -------------
all_tokens = [t for s in spr_orig["train"]["sequence"] for t in s.split()]
shape_chars = sorted({tok[0] for tok in all_tokens})
color_chars = sorted({tok[1] for tok in all_tokens})


def build_perm(xs):
    shuffled = xs[:]
    random.shuffle(shuffled)
    return {a: b for a, b in zip(xs, shuffled)}


# token-renamed : random permutation of BOTH shape + colour chars
shape_ren_map, color_ren_map = build_perm(shape_chars), build_perm(color_chars)
# colour-shuffled : identity for shape, permutation only for colour
shape_identity = {c: c for c in shape_chars}
color_shuffle_map = build_perm(color_chars)


def transform_dataset(base: DatasetDict, s_map, c_map) -> DatasetDict:
    def _tr(example):
        def convert(tok):
            # keep length-2 tokens, otherwise leave untouched
            return (
                s_map.get(tok[0], tok[0]) + c_map.get(tok[1], tok[1])
                if len(tok) == 2
                else tok
            )

        example["sequence"] = " ".join(convert(t) for t in example["sequence"].split())
        return example

    out = {}
    for split in ["train", "dev", "test"]:
        out[split] = base[split].map(
            _tr, load_from_cache_file=False, desc=f"building split {split}"
        )
    return DatasetDict(out)


spr_renamed = transform_dataset(spr_orig, shape_ren_map, color_ren_map)
spr_cshuffled = transform_dataset(spr_orig, shape_identity, color_shuffle_map)
print("Transformed datasets ready.")


# ----------------- symbols / metrics ----------------------
def count_shape_variety(seq: str):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    good = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(good) / max(sum(w), 1e-8)


# ----------------- vocabulary -----------------------------
class Vocab:
    def __init__(self, toks):
        self.itos = ["<pad>", "<unk>"] + sorted(set(toks))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks]

    def __len__(self):
        return len(self.itos)


# build joint vocabulary from all three training sets
joint_tokens = []
for ds in [spr_orig, spr_renamed, spr_cshuffled]:
    joint_tokens += [tok for seq in ds["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(joint_tokens)

labels = sorted(set(spr_orig["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

MAX_LEN = 50


def seq_to_tensor(seq):
    ids = vocab.encode(seq.split()[:MAX_LEN])
    pad = [0] * (MAX_LEN - len(ids))
    mask = [1] * len(ids) + [0] * len(pad)
    return ids + pad, mask


def collate(batch):
    tok_mat, mask_mat, symb_feats, labs, seqs = [], [], [], [], []
    for ex in batch:
        ids, msk = seq_to_tensor(ex["sequence"])
        sv = count_shape_variety(ex["sequence"])
        cv = count_color_variety(ex["sequence"])
        ln = len(ex["sequence"].split())
        symb = [sv, cv, ln, sv / (ln + 1e-6), cv / (ln + 1e-6)]
        tok_mat.append(ids)
        mask_mat.append(msk)
        symb_feats.append(symb)
        labs.append(label2id[ex["label"]])
        seqs.append(ex["sequence"])
    return (
        torch.tensor(tok_mat, device=device),
        torch.tensor(mask_mat, device=device, dtype=torch.bool),
        torch.tensor(symb_feats, device=device, dtype=torch.float32),
        torch.tensor(labs, device=device),
        seqs,
    )


# ------------------ model ---------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NeuroSymbolicSPR(nn.Module):
    def __init__(self, vocab_sz, emb_dim, n_heads, n_layers, symb_dim, n_cls):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim)
        enc = nn.TransformerEncoderLayer(
            emb_dim, n_heads, emb_dim * 2, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim + symb_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )
        self.sv = nn.Linear(emb_dim, 1)
        self.cv = nn.Linear(emb_dim, 1)

    def forward(self, tok, mask, symb):
        emb = self.embedding(tok)
        emb = self.pos(emb)
        enc = self.encoder(emb, src_key_padding_mask=~mask.bool())
        pooled = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        logit = self.cls_head(torch.cat([pooled, symb], -1))
        return logit, self.sv(pooled).squeeze(-1), self.cv(pooled).squeeze(-1)


# ------------------ dataloaders ---------------------------
batch_size = 256
train_concat = concatenate_datasets(
    [spr_orig["train"], spr_renamed["train"], spr_cshuffled["train"]]
)
train_loader = DataLoader(
    train_concat, batch_size=batch_size, shuffle=True, collate_fn=collate
)

dev_loaders = {
    "SPR_BENCH": DataLoader(
        spr_orig["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
    ),
    "TOKEN_RENAMED": DataLoader(
        spr_renamed["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
    ),
    "COLOR_SHUFFLED": DataLoader(
        spr_cshuffled["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
    ),
}

test_loaders = {
    "SPR_BENCH": DataLoader(
        spr_orig["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
    ),
    "TOKEN_RENAMED": DataLoader(
        spr_renamed["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
    ),
    "COLOR_SHUFFLED": DataLoader(
        spr_cshuffled["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
    ),
}

# ------------------ training objects ----------------------
model = NeuroSymbolicSPR(
    len(vocab), 64, n_heads=4, n_layers=2, symb_dim=5, n_cls=len(labels)
).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------ training loop -------------------------
best_avg_swa, patience, wait, epochs = 0.0, 3, 0, 20
for ep in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    for tok, msk, symb, lab, _ in train_loader:
        optimizer.zero_grad()
        logits, sv_pred, cv_pred = model(tok, msk, symb)
        sv_true = symb[:, 0]
        cv_true = symb[:, 1]
        loss = (
            criterion_cls(logits, lab)
            + 0.2 * criterion_reg(sv_pred, sv_true)
            + 0.2 * criterion_reg(cv_pred, cv_true)
        )
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * lab.size(0)
    tr_loss /= len(train_concat)

    # ------------ validation on three dev sets ------------
    val_swas = []
    model.eval()
    for name, loader in dev_loaders.items():
        y_t, y_p, seqs = [], [], []
        vloss = 0.0
        with torch.no_grad():
            for tok, msk, symb, lab, seq in loader:
                logits, sv_p, cv_p = model(tok, msk, symb)
                sv_t = symb[:, 0]
                cv_t = symb[:, 1]
                vloss += (
                    criterion_cls(logits, lab)
                    + 0.2 * criterion_reg(sv_p, sv_t)
                    + 0.2 * criterion_reg(cv_p, cv_t)
                ).item() * lab.size(0)
                preds = logits.argmax(1).cpu().tolist()
                y_p.extend([id2label[p] for p in preds])
                y_t.extend([id2label[i] for i in lab.cpu().tolist()])
                seqs.extend(seq)
        vloss /= len(loader.dataset)
        swa = shape_weighted_accuracy(seqs, y_t, y_p)
        experiment_data["MSDT"][name]["losses"]["val"].append(vloss)
        experiment_data["MSDT"][name]["metrics"]["val"].append(swa)
        val_swas.append(swa)
    # record global train loss to each dataset entry for convenience
    for name in experiment_data["MSDT"]:
        if name == "SPR_BENCH" or name == "TOKEN_RENAMED" or name == "COLOR_SHUFFLED":
            experiment_data["MSDT"][name]["losses"]["train"].append(tr_loss)
            experiment_data["MSDT"][name]["metrics"]["train"].append(None)

    avg_swa = sum(val_swas) / len(val_swas)
    print(
        f"Epoch {ep} | train_loss={tr_loss:.4f} | dev SWAs={val_swas} | avg={avg_swa:.4f}"
    )
    if avg_swa > best_avg_swa:
        best_avg_swa = avg_swa
        wait = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ------------------ test evaluation -----------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
for name, loader in test_loaders.items():
    y_t, y_p, seqs = [], [], []
    with torch.no_grad():
        for tok, msk, symb, lab, seq in loader:
            logits, _, _ = model(tok, msk, symb)
            preds = logits.argmax(1).cpu().tolist()
            y_p.extend([id2label[p] for p in preds])
            y_t.extend([id2label[i] for i in lab.cpu().tolist()])
            seqs.extend(seq)
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    experiment_data["MSDT"][name]["predictions"] = y_p
    experiment_data["MSDT"][name]["ground_truth"] = y_t
    print(f"TEST {name} SWA: {swa:.4f}")

# ------------------ save ----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
