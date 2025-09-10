import os, pathlib, warnings, random, string, math, time
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# --------------------- working dir & device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- SPR helpers (with fallback) ---------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy
except Exception as e:
    warnings.warn(f"SPR helpers not found, using fallbacks ({e})")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError

    def count_shape_variety(sequence: str) -> int:
        return len(set(tok[0] for tok in sequence.strip().split() if tok))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [count_shape_variety(s) for s in seqs]
        c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(c) / (sum(w) + 1e-9)


# --------------------- synthetic fallback ------------------------------
def make_synthetic_dataset(n):
    shapes = list(string.ascii_uppercase[:6])
    colors = list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(4, 10)
        tokens = [random.choice(shapes) + random.choice(colors) for _ in range(L)]
        seqs.append(" ".join(tokens))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# --------------------- load dataset ------------------------------------
root_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
try:
    dsets = load_spr_bench(root_path)
    print("Loaded official SPR_BENCH.")
    train_seqs, train_labels = dsets["train"]["sequence"], dsets["train"]["label"]
    dev_seqs, dev_labels = dsets["dev"]["sequence"], dsets["dev"]["label"]
    test_seqs, test_labels = dsets["test"]["sequence"], dsets["test"]["label"]
except Exception as e:
    warnings.warn(f"Cannot load SPR_BENCH: {e}\nUsing synthetic data.")
    train = make_synthetic_dataset(800)
    dev = make_synthetic_dataset(200)
    test = make_synthetic_dataset(400)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]

num_classes = int(max(train_labels + dev_labels + test_labels)) + 1
print(f"Classes: {num_classes}")

# --------------------- tokenisation ------------------------------------
PAD = "<PAD>"
all_tokens = set(tok for seq in train_seqs for tok in seq.split())
tok2id = {PAD: 0}
for t in sorted(all_tokens):
    tok2id[t] = len(tok2id)
id2tok = {i: t for t, i in tok2id.items()}
pad_id = tok2id[PAD]
max_len = max(len(s.split()) for s in train_seqs + dev_seqs + test_seqs)


def encode_seq(seq):
    ids = [tok2id.get(t, 0) for t in seq.split()[:max_len]]
    ids = ids + [pad_id] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def symbolic_feat(seq):
    toks = seq.split()
    shapes = set([t[0] for t in toks if t])
    colors = set([t[1] for t in toks if len(t) > 1])
    return np.array([len(shapes), len(colors), len(toks)], dtype=np.float32)


class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seq_enc = np.stack([encode_seq(s) for s in seqs])
        self.symb = np.stack([symbolic_feat(s) for s in seqs])
        self.y = np.array(labels, dtype=np.int64)
        self.raw_seqs = seqs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "ids": torch.from_numpy(self.seq_enc[idx]),
            "sym": torch.from_numpy(self.symb[idx]),
            "y": torch.tensor(self.y[idx]),
            "raw": self.raw_seqs[idx],
        }


train_ds, dev_ds, test_ds = (
    SPRDataset(train_seqs, train_labels),
    SPRDataset(dev_seqs, dev_labels),
    SPRDataset(test_seqs, test_labels),
)
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# --------------------- model -------------------------------------------
class HybridModel(nn.Module):
    def __init__(
        self,
        vocab_sz,
        embed_dim=64,
        nhead=4,
        num_layers=2,
        sym_feat=3,
        num_classes=4,
        max_len=50,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=128
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + sym_feat, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, ids, sym):
        B, L = ids.shape
        tok = self.tok_emb(ids)
        pos_ids = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        tok = tok + self.pos_emb(pos_ids)
        tok = tok.transpose(0, 1)  # Transformer expects (L,B,E)
        enc = self.transformer(tok)  # (L,B,E)
        cls = enc[0]  # take first position
        out = self.classifier(torch.cat([cls, sym], dim=1))
        return out


model = HybridModel(len(tok2id), max_len=max_len, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# --------------------- experiment tracking -----------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_SWA": [], "dev_SWA": []},
        "losses": {"train": [], "dev": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": test_labels,
        "test_metrics": {},
    }
}

# --------------------- training loop -----------------------------------
max_epochs = 30
patience = 5
best_dev_swa = -1.0
wait = 0
for epoch in range(1, max_epochs + 1):
    # train
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        ids = batch["ids"].to(device)
        sym = batch["sym"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits = model(ids, sym)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * ids.size(0)
    tr_loss /= len(train_ds)
    # compute train SWA
    with torch.no_grad():
        logits = model(
            torch.from_numpy(train_ds.seq_enc).to(device),
            torch.from_numpy(train_ds.symb).to(device),
        )
        train_pred = logits.argmax(1).cpu().numpy()
    train_swa = shape_weighted_accuracy(train_seqs, train_labels, train_pred)
    # validate
    model.eval()
    dev_loss = 0.0
    dev_preds = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["ids"].to(device)
            sym = batch["sym"].to(device)
            y = batch["y"].to(device)
            logits = model(ids, sym)
            dev_loss += criterion(logits, y).item() * ids.size(0)
            dev_preds.append(logits.cpu())
    dev_loss /= len(dev_ds)
    dev_pred = torch.cat(dev_preds).argmax(1).numpy()
    dev_swa = shape_weighted_accuracy(dev_seqs, dev_labels, dev_pred)
    # log
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["dev"].append(dev_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_SWA"].append(train_swa)
    experiment_data["SPR_BENCH"]["metrics"]["dev_SWA"].append(dev_swa)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} dev_loss={dev_loss:.4f} dev_SWA={dev_swa:.4f}"
    )
    # early stopping
    if dev_swa > best_dev_swa + 1e-5:
        best_dev_swa = dev_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# restore best
model.load_state_dict(best_state)

# --------------------- test evaluation ---------------------------------
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        ids = batch["ids"].to(device)
        sym = batch["sym"].to(device)
        test_preds.append(model(ids, sym).cpu())
test_pred = torch.cat(test_preds).argmax(1).numpy()
test_swa = shape_weighted_accuracy(test_seqs, test_labels, test_pred)
print(f"\nTest Shape-Weighted Accuracy (SWA): {test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["test_metrics"] = {"SWA": test_swa}

# --------------------- save artefacts ----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Artifacts saved to ./working")
