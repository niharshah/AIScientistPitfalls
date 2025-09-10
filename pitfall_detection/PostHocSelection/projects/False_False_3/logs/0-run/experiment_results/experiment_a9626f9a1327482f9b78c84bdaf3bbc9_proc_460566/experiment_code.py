import os, pathlib, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------- house-keeping ---------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# container for all logs
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": [], "train_rcaa": [], "val_rcaa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------------------- data utilities ------------------------------------
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


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) if sum(weights) > 0 else 1e-8)


def rule_complexity_adjusted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) if sum(weights) > 0 else 1e-8)


# ---------------------------------- vocab -------------------------------------------
class Vocab:
    def __init__(self, tokens):
        self.itos = ["<pad>", "<unk>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(tok, 1) for tok in toks]


# --------------------------------- model --------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NeuroSymbolicSPR(nn.Module):
    def __init__(self, vocab_sz, emb_dim, n_heads, n_layers, n_cls):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.posenc = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            emb_dim, n_heads, dim_feedforward=emb_dim * 2, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, tok_mat, mask):
        x = self.embedding(tok_mat)
        x = self.posenc(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        logits = self.cls_head(pooled)
        return logits


# ----------------------------- load and prepare data ---------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

vocab = Vocab([tok for seq in spr["train"]["sequence"] for tok in seq.split()])
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


def collate(batch):
    sequences = [ex["sequence"] for ex in batch]
    tokens = [seq.split() for seq in sequences]
    max_len = max(len(t) for t in tokens)
    ids, masks, labs = [], [], []
    for toks, ex in zip(tokens, batch):
        enc = vocab.encode(toks)
        pad_len = max_len - len(enc)
        ids.append(enc + [0] * pad_len)
        masks.append([1] * len(enc) + [0] * pad_len)
        labs.append(label2id[ex["label"]])
    return (
        torch.tensor(ids, device=device),
        torch.tensor(masks, device=device, dtype=torch.bool),
        torch.tensor(labs, device=device),
        sequences,
    )


train_loader = DataLoader(
    spr["train"], batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(spr["dev"], batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(spr["test"], batch_size=256, shuffle=False, collate_fn=collate)

# -------------------------------- training stuff -------------------------------------
model = NeuroSymbolicSPR(len(vocab), 64, n_heads=4, n_layers=2, n_cls=len(labels)).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_swa, patience, wait = 0.0, 3, 0
epochs = 20

for epoch in range(1, epochs + 1):
    # ----- training -----
    model.train()
    epoch_loss, y_pred_tr, y_true_tr, seq_tr = 0.0, [], [], []
    for tok, msk, lab, seqs in train_loader:
        optimizer.zero_grad()
        logits = model(tok, msk)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * lab.size(0)
        preds = logits.argmax(1).cpu().tolist()
        y_pred_tr.extend([id2label[p] for p in preds])
        y_true_tr.extend([id2label[i] for i in lab.cpu().tolist()])
        seq_tr.extend(seqs)
    epoch_loss /= len(spr["train"])
    train_swa = shape_weighted_accuracy(seq_tr, y_true_tr, y_pred_tr)
    train_rcaa = rule_complexity_adjusted_accuracy(seq_tr, y_true_tr, y_pred_tr)

    # ----- validation -----
    model.eval()
    val_loss, y_true_v, y_pred_v, seq_v = 0.0, [], [], []
    with torch.no_grad():
        for tok, msk, lab, seqs in dev_loader:
            logits = model(tok, msk)
            loss = criterion(logits, lab)
            val_loss += loss.item() * lab.size(0)
            preds = logits.argmax(1).cpu().tolist()
            y_pred_v.extend([id2label[p] for p in preds])
            y_true_v.extend([id2label[i] for i in lab.cpu().tolist()])
            seq_v.extend(seqs)
    val_loss /= len(spr["dev"])
    val_swa = shape_weighted_accuracy(seq_v, y_true_v, y_pred_v)
    val_rcaa = rule_complexity_adjusted_accuracy(seq_v, y_true_v, y_pred_v)

    print(
        f"Epoch {epoch}: train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | "
        f"SWA={val_swa:.4f} | RCAA={val_rcaa:.4f}"
    )

    # store logs
    ed = experiment_data["SPR_BENCH"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(epoch_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_swa"].append(train_swa)
    ed["metrics"]["val_swa"].append(val_swa)
    ed["metrics"]["train_rcaa"].append(train_rcaa)
    ed["metrics"]["val_rcaa"].append(val_rcaa)

    if val_swa > best_val_swa:
        best_val_swa = val_swa
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ------------------------------ test evaluation --------------------------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
y_true_te, y_pred_te, seq_te = [], [], []
with torch.no_grad():
    for tok, msk, lab, seqs in test_loader:
        logits = model(tok, msk)
        preds = logits.argmax(1).cpu().tolist()
        y_pred_te.extend([id2label[p] for p in preds])
        y_true_te.extend([id2label[i] for i in lab.cpu().tolist()])
        seq_te.extend(seqs)
test_swa = shape_weighted_accuracy(seq_te, y_true_te, y_pred_te)
test_rcaa = rule_complexity_adjusted_accuracy(seq_te, y_true_te, y_pred_te)
print(f"Test SWA: {test_swa:.4f} | Test RCAA: {test_rcaa:.4f}")

ed = experiment_data["SPR_BENCH"]
ed["predictions"] = y_pred_te
ed["ground_truth"] = y_true_te
ed["test_swa"] = test_swa
ed["test_rcaa"] = test_rcaa

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
