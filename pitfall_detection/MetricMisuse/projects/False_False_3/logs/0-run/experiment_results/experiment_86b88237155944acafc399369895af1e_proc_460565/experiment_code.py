# No-Positional-Encoding (No-PE) ablation for SPR-Bench
import os, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ------------- boilerplate -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- data utils -------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    good = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(good) / max(sum(w), 1e-8)


# ------------- vocab ------------------------------------------------------------------
class Vocab:
    def __init__(self, tokens):
        self.itos = ["<pad>", "<unk>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks]

    def __len__(self):
        return len(self.itos)


# ------------- model (No-PE) ----------------------------------------------------------
class NeuroSymbolicSPR_NoPE(nn.Module):
    """
    Identical to baseline except NO positional encoding is added to token embeddings.
    """

    def __init__(self, vocab_sz, emb_dim, n_heads, n_layers, symb_dim, n_cls):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            emb_dim, n_heads, emb_dim * 2, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim + symb_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )
        self.sv_head = nn.Linear(emb_dim, 1)
        self.cv_head = nn.Linear(emb_dim, 1)

    def forward(self, tok_mat, mask, symb_feats):
        emb = self.embedding(tok_mat)  # (B,L,D)
        # NO positional encoding here
        enc_out = self.encoder(emb, src_key_padding_mask=~mask.bool())
        pooled = (enc_out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        cls_in = torch.cat([pooled, symb_feats], -1)
        return (
            self.cls_head(cls_in),
            self.sv_head(pooled).squeeze(-1),
            self.cv_head(pooled).squeeze(-1),
        )


# ------------- load dataset -----------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

MAX_LEN = 50  # >99% sequences


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

# ------------- experiment record ------------------------------------------------------
experiment_data = {
    "No-PE": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------- instantiate model ------------------------------------------------------
model = NeuroSymbolicSPR_NoPE(
    len(vocab), 64, n_heads=4, n_layers=2, symb_dim=5, n_cls=len(labels)
).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------- training loop ----------------------------------------------------------
best_swa, patience, wait, epochs = 0.0, 3, 0, 20
for epoch in range(1, epochs + 1):
    # ---- train -----------------------------------------------------------------------
    model.train()
    tr_loss = 0.0
    for tok, msk, symb, lab, _ in train_loader:
        optimizer.zero_grad()
        logits, sv_pred, cv_pred = model(tok, msk, symb)
        sv_true, cv_true = symb[:, 0], symb[:, 1]
        loss = (
            criterion_cls(logits, lab)
            + 0.2 * criterion_reg(sv_pred, sv_true)
            + 0.2 * criterion_reg(cv_pred, cv_true)
        )
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * lab.size(0)
    tr_loss /= len(spr["train"])

    # ---- validate --------------------------------------------------------------------
    model.eval()
    val_loss, y_true, y_pred, sequences = 0.0, [], [], []
    with torch.no_grad():
        for tok, msk, symb, lab, seq in dev_loader:
            logits, sv_p, cv_p = model(tok, msk, symb)
            sv_true, cv_true = symb[:, 0], symb[:, 1]
            vloss = (
                criterion_cls(logits, lab)
                + 0.2 * criterion_reg(sv_p, sv_true)
                + 0.2 * criterion_reg(cv_p, cv_true)
            )
            val_loss += vloss.item() * lab.size(0)
            preds = logits.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in lab.cpu().tolist()])
            sequences.extend(seq)
    val_loss /= len(spr["dev"])
    val_swa = shape_weighted_accuracy(sequences, y_true, y_pred)

    # record
    experiment_data["No-PE"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["No-PE"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["No-PE"]["SPR_BENCH"]["metrics"]["train"].append(None)
    experiment_data["No-PE"]["SPR_BENCH"]["metrics"]["val"].append(val_swa)

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | SWA={val_swa:.4f}"
    )

    # early stop
    if val_swa > best_swa:
        best_swa = val_swa
        torch.save(model.state_dict(), os.path.join(working_dir, "best_nope.pt"))
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ------------- test evaluation --------------------------------------------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best_nope.pt")))
model.eval()
y_true, y_pred, sequences = [], [], []
with torch.no_grad():
    for tok, msk, symb, lab, seq in test_loader:
        logits, _, _ = model(tok, msk, symb)
        preds = logits.argmax(1).cpu().tolist()
        y_pred.extend([id2label[p] for p in preds])
        y_true.extend([id2label[i] for i in lab.cpu().tolist()])
        sequences.extend(seq)
test_swa = shape_weighted_accuracy(sequences, y_true, y_pred)
print(f"Test Shape-Weighted Accuracy (SWA) [No-PE]: {test_swa:.4f}")

# store predictions / truths
experiment_data["No-PE"]["SPR_BENCH"]["predictions"] = y_pred
experiment_data["No-PE"]["SPR_BENCH"]["ground_truth"] = y_true

# ------------- save experiment --------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
