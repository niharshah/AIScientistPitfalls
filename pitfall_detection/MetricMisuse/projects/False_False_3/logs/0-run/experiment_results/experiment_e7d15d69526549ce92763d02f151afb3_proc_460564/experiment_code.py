import os, math, time, random, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------ paths / GPU
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------ experiment log
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},  # we store tuples (SWA,CWA,RCAA)
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------ dataset utils
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):  # treat each csv as its own split
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    good = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(good) / (sum(w) + 1e-8)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    good = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(good) / (sum(w) + 1e-8)


def rcaa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    good = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(good) / (sum(w) + 1e-8)


# ------------------------------------------------------------------ vocab
class Vocab:
    def __init__(self, tokens):
        special = ["<pad>", "<unk>"]
        self.itos = special + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks):
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in toks]

    def __len__(self):
        return len(self.itos)


# ------------------------------------------------------------------ model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NeuroSymbolicSPR(nn.Module):
    def __init__(self, vocab_sz, emb_dim, n_heads, n_layers, symb_dim, n_cls):
        super().__init__()
        self.symb_dim = symb_dim
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.posenc = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            emb_dim, n_heads, dim_feedforward=emb_dim * 2, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim + symb_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )
        # auxiliary regression heads
        self.sv_head = nn.Linear(emb_dim, 1)
        self.cv_head = nn.Linear(emb_dim, 1)

    def forward(self, tok_mat, attn_mask, symb_feats):
        emb = self.embedding(tok_mat)
        emb = self.posenc(emb)
        enc_out = self.encoder(emb, src_key_padding_mask=~attn_mask.bool())
        pooled = (enc_out * attn_mask.unsqueeze(-1).float()).sum(1) / attn_mask.sum(
            1, keepdim=True
        ).clamp(min=1).float()
        if self.symb_dim > 0:
            cls_in = torch.cat([pooled, symb_feats], -1)
        else:
            cls_in = pooled
        logits = self.cls_head(cls_in)
        sv_pred = self.sv_head(pooled).squeeze(-1)
        cv_pred = self.cv_head(pooled).squeeze(-1)
        return logits, sv_pred, cv_pred


# ------------------------------------------------------------------ data prep
DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

token_bank = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(token_bank)

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


def collate(batch, max_seq_cap=256):
    ids_list, seqs = [], []
    svs, cvs, lens = [], [], []
    labs = []
    for ex in batch:
        toks = ex["sequence"].split()
        ids = vocab.encode(toks[:max_seq_cap])  # truncate only if >cap
        ids_list.append(ids)
        seqs.append(ex["sequence"])
        sv, cv = count_shape_variety(ex["sequence"]), count_color_variety(
            ex["sequence"]
        )
        svs.append(sv)
        cvs.append(cv)
        lens.append(len(toks))
        labs.append(label2id[ex["label"]])
    max_len = max(len(ids) for ids in ids_list)
    tok_mat, msk_mat = [], []
    for ids in ids_list:
        pad = [0] * (max_len - len(ids))
        tok_mat.append(ids + pad)
        msk_mat.append([1] * len(ids) + [0] * len(pad))
    symb_feats = np.stack(
        [
            svs,
            cvs,
            lens,
            np.array(svs) / (np.array(lens) + 1e-6),
            np.array(cvs) / (np.array(lens) + 1e-6),
        ],
        1,
    )
    return (
        torch.tensor(tok_mat, dtype=torch.long).to(device),
        torch.tensor(msk_mat, dtype=torch.bool).to(device),
        torch.tensor(symb_feats, dtype=torch.float32).to(device),
        torch.tensor(labs, dtype=torch.long).to(device),
        seqs,
    )


batch_size = 256
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ------------------------------------------------------------------ instantiate + optim
model = NeuroSymbolicSPR(
    vocab_sz=len(vocab),
    emb_dim=64,
    n_heads=4,
    n_layers=2,
    symb_dim=5,
    n_cls=len(labels),
).to(device)

criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------ train
best_val_rcaa, patience, stall = 0.0, 3, 0
epochs = 15
for epoch in range(1, epochs + 1):
    # ---------- train ----------
    model.train()
    t_loss = 0.0
    for tok, mask, symb, lab, _ in train_loader:
        optimizer.zero_grad()
        logits, sv_p, cv_p = model(tok, mask, symb)
        sv_true, cv_true = symb[:, 0], symb[:, 1]
        loss = (
            criterion_cls(logits, lab)
            + 0.2 * criterion_reg(sv_p, sv_true)
            + 0.2 * criterion_reg(cv_p, cv_true)
        )
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * lab.size(0)
    t_loss /= len(spr["train"])

    # ---------- validate ----------
    model.eval()
    v_loss, y_true, y_pred, seq_bank = 0.0, [], [], []
    with torch.no_grad():
        for tok, mask, symb, lab, seqs in dev_loader:
            logits, sv_p, cv_p = model(tok, mask, symb)
            sv_true, cv_true = symb[:, 0], symb[:, 1]
            loss = (
                criterion_cls(logits, lab)
                + 0.2 * criterion_reg(sv_p, sv_true)
                + 0.2 * criterion_reg(cv_p, cv_true)
            )
            v_loss += loss.item() * lab.size(0)
            preds = logits.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in lab.cpu().tolist()])
            seq_bank.extend(seqs)
    v_loss /= len(spr["dev"])
    val_swa = shape_weighted_accuracy(seq_bank, y_true, y_pred)
    val_cwa = color_weighted_accuracy(seq_bank, y_true, y_pred)
    val_rcaa = rcaa(seq_bank, y_true, y_pred)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(t_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((val_swa, val_cwa, val_rcaa))

    print(
        f"Epoch {epoch}: val_loss={v_loss:.4f} | "
        f"SWA={val_swa:.3f} | CWA={val_cwa:.3f} | RCAA={val_rcaa:.3f}"
    )

    # ---------- early-stop on RCAA ----------
    if val_rcaa > best_val_rcaa:
        best_val_rcaa = val_rcaa
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
        stall = 0
    else:
        stall += 1
        if stall >= patience:
            print("Early stopping.")
            break

# ------------------------------------------------------------------ test evaluation
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
y_true, y_pred, seq_bank = [], [], []
with torch.no_grad():
    for tok, mask, symb, lab, seqs in test_loader:
        logits, _, _ = model(tok, mask, symb)
        preds = logits.argmax(1).cpu().tolist()
        y_pred.extend([id2label[p] for p in preds])
        y_true.extend([id2label[i] for i in lab.cpu().tolist()])
        seq_bank.extend(seqs)
test_swa = shape_weighted_accuracy(seq_bank, y_true, y_pred)
test_cwa = color_weighted_accuracy(seq_bank, y_true, y_pred)
test_rcaa = rcaa(seq_bank, y_true, y_pred)
print(f"TEST >> SWA={test_swa:.3f} | CWA={test_cwa:.3f} | RCAA={test_rcaa:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
