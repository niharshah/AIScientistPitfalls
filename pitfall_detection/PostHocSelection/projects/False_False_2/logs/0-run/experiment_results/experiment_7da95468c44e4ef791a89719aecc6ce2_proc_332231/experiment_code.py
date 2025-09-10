import os, pathlib, warnings, random, string, time
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------- housekeeping & device -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------- import benchmark helpers ----------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy

    BENCH_AVAILABLE = True
except Exception as e:
    warnings.warn(f"Could not import SPR helpers ({e}); synthetic fall-back used.")
    BENCH_AVAILABLE = False

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError

    def _count_shape_variety(seq: str):
        return len(set(tok[0] for tok in seq.split() if tok))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_shape_variety(s) for s in seqs]
        return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / (
            sum(w) + 1e-9
        )


# ------------------- synthetic data helper --------------------------------
def make_synthetic(n: int):
    shapes, cols = list(string.ascii_uppercase[:6]), list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n):
        ln = random.randint(4, 9)
        seqs.append(
            " ".join(random.choice(shapes) + random.choice(cols) for _ in range(ln))
        )
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ------------------- load data -------------------------------------------
root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dsets = load_spr_bench(root)
    train_raw, dev_raw, test_raw = dsets["train"], dsets["dev"], dsets["test"]
    train_seqs, train_lbl = train_raw["sequence"], train_raw["label"]
    dev_seqs, dev_lbl = dev_raw["sequence"], dev_raw["label"]
    test_seqs, test_lbl = test_raw["sequence"], test_raw["label"]
    print("Loaded real SPR_BENCH.")
except Exception as e:
    warnings.warn(f"{e}\nUsing synthetic data.")
    train, dev, test = make_synthetic(4000), make_synthetic(800), make_synthetic(1600)
    train_seqs, train_lbl = train["sequence"], train["label"]
    dev_seqs, dev_lbl = dev["sequence"], dev["label"]
    test_seqs, test_lbl = test["sequence"], test["label"]

n_classes = int(max(train_lbl + dev_lbl + test_lbl)) + 1
print(f"Classes: {n_classes}")

# ------------------- vocabulary & encoders -------------------------------
all_tokens = {tok for seq in train_seqs for tok in seq.split()}
PAD, UNK = "<PAD>", "<UNK>"
tok2idx = {PAD: 0, UNK: 1, **{t: i + 2 for i, t in enumerate(sorted(all_tokens))}}
idx2tok = {i: t for t, i in tok2idx.items()}
max_len = max(len(s.split()) for s in train_seqs + dev_seqs + test_seqs)
print(f"Vocabulary size: {len(tok2idx)} | max_len: {max_len}")


def encode_seq(seq: str):
    ids = [tok2idx.get(t, tok2idx[UNK]) for t in seq.split()]
    ids = ids[:max_len] + [tok2idx[PAD]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def sym_feats(seq: str):
    toks = seq.split()
    return np.array(
        [
            len(set(t[0] for t in toks)),
            len(set(t[1] for t in toks if len(t) > 1)),
            len(toks),
        ],
        dtype=np.float32,
    )


# ------------------- Dataset ---------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, sequences, labels, zscore=None):
        self.X_tok = np.stack([encode_seq(s) for s in sequences])
        raw_sym = np.stack([sym_feats(s) for s in sequences])
        if zscore is not None:
            mu, sig = zscore
            raw_sym = (raw_sym - mu) / sig
        self.X_sym = raw_sym
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_tok[idx], dtype=torch.long),
            torch.tensor(self.X_sym[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


train_sym_raw = np.stack([sym_feats(s) for s in train_seqs])
mu, sig = train_sym_raw.mean(0), train_sym_raw.std(0) + 1e-6
train_ds = SPRDataset(train_seqs, train_lbl, (mu, sig))
dev_ds = SPRDataset(dev_seqs, dev_lbl, (mu, sig))
test_ds = SPRDataset(test_seqs, test_lbl, (mu, sig))

batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)


# ------------------- Model --------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class NeuroSymbolicModel(nn.Module):
    def __init__(
        self, vocab_size: int, emb_dim=32, sym_dim=3, num_classes=4, use_sym=True
    ):
        super().__init__()
        self.use_sym = use_sym
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.posenc = PositionalEncoding(emb_dim, max_len)

        in_dim = emb_dim + sym_dim if use_sym else emb_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, tok_ids, sym_feats):
        # tok_ids : [B,L]  sym_feats : [B,3]
        emb = self.embed(tok_ids).transpose(0, 1)  # [L,B,E]
        emb = self.posenc(emb)
        enc = self.encoder(emb).mean(0)  # [B,E]

        if self.use_sym:
            z = torch.cat([enc, sym_feats], dim=1)
        else:
            z = enc
        return self.head(z)


# ------------------- training utilities ----------------------------------
def run_epoch(model, loader, optimizer=None):
    train_flag = optimizer is not None
    criterion = nn.CrossEntropyLoss()
    total_loss, preds, gts = 0.0, [], []
    model.train() if train_flag else model.eval()

    for tok, sym, y in loader:
        tok, sym, y = tok.to(device), sym.to(device), y.to(device)
        out = model(tok, sym)
        loss = criterion(out, y)
        if train_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
        preds.append(out.argmax(1).cpu().numpy())
        gts.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return total_loss / len(loader.dataset), preds, gts


def evaluate_SWA(seqs, y_true, y_pred):
    return shape_weighted_accuracy(seqs, y_true, y_pred)


# ------------------- experiment logging ----------------------------------
experiment_data = {
    "spr_neuro_sym": {
        "metrics": {"train_SWA": [], "dev_SWA": []},
        "losses": {"train": [], "dev": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "test_SWA": None,
        "timestamps": [],
    }
}

# ------------------- full model training ---------------------------------
full_model = NeuroSymbolicModel(
    vocab_size=len(tok2idx), num_classes=n_classes, use_sym=True
).to(device)
optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-3, weight_decay=1e-4)

best_dev, patience, wait = -1.0, 6, 0
max_epochs = 30
best_state = None

for epoch in range(1, max_epochs + 1):
    tr_loss, tr_pred, tr_gt = run_epoch(full_model, train_dl, optimizer)
    dv_loss, dv_pred, dv_gt = run_epoch(full_model, dev_dl)

    tr_swa = evaluate_SWA(train_seqs, train_lbl, tr_pred)
    dv_swa = evaluate_SWA(dev_seqs, dev_lbl, dv_pred)

    log = experiment_data["spr_neuro_sym"]
    log["epochs"].append(epoch)
    log["losses"]["train"].append(tr_loss)
    log["losses"]["dev"].append(dv_loss)
    log["metrics"]["train_SWA"].append(tr_swa)
    log["metrics"]["dev_SWA"].append(dv_swa)
    log["timestamps"].append(datetime.utcnow().isoformat())

    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f}  dev_loss={dv_loss:.4f}  "
        f"dev_SWA={dv_swa:.4f}"
    )

    if dv_swa > best_dev + 1e-5:
        best_dev = dv_swa
        best_state = {k: v.cpu() for k, v in full_model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stop triggered.")
            break

full_model.load_state_dict(best_state)

# ------------------- Test evaluation -------------------------------------
_, tst_pred, _ = run_epoch(full_model, test_dl)
test_swa = evaluate_SWA(test_seqs, test_lbl, tst_pred)
print(f"\nTEST SWA (full model) = {test_swa:.4f}")

experiment_data["spr_neuro_sym"]["predictions"] = tst_pred
experiment_data["spr_neuro_sym"]["ground_truth"] = np.array(test_lbl)
experiment_data["spr_neuro_sym"]["test_SWA"] = test_swa

# ------------------- Ablation (no symbolic features) ----------------------
ablation_model = NeuroSymbolicModel(
    vocab_size=len(tok2idx), num_classes=n_classes, use_sym=False
).to(device)
opt2 = torch.optim.Adam(ablation_model.parameters(), lr=1e-3, weight_decay=1e-4)

best_dev2, wait = -1.0, 0
for epoch in range(1, max_epochs + 1):
    run_epoch(ablation_model, train_dl, opt2)
    dv_loss2, dv_pred2, _ = run_epoch(ablation_model, dev_dl)
    dv_swa2 = evaluate_SWA(dev_seqs, dev_lbl, dv_pred2)
    if dv_swa2 > best_dev2 + 1e-5:
        best_dev2 = dv_swa2
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break
print(
    f"Ablation (no sym) best dev SWA: {best_dev2:.4f}  |  Full model best dev SWA: {best_dev:.4f}"
)

# ------------------- Save artefacts ---------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure()
plt.plot(
    experiment_data["spr_neuro_sym"]["epochs"],
    experiment_data["spr_neuro_sym"]["losses"]["train"],
    label="train",
)
plt.plot(
    experiment_data["spr_neuro_sym"]["epochs"],
    experiment_data["spr_neuro_sym"]["losses"]["dev"],
    label="dev",
)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.title("Loss curve – SPR")
plt.savefig(os.path.join(working_dir, "loss_curve_SPR.png"))
plt.close()

print("Finished. Artefacts saved in ./working")
