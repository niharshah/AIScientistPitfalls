import os, pathlib, warnings, string, random, time
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- basic working dir & device --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- try to import official helpers ----------------------
try:
    from SPR import (
        load_spr_bench,
        shape_weighted_accuracy,
        count_shape_variety,
        count_color_variety,
    )

    HAVE_SPR = True
except Exception as e:
    warnings.warn(f"Could not import official SPR utils: {e}")
    HAVE_SPR = False

    # minimal fall-backs
    def count_shape_variety(seq):
        return len(set(tok[0] for tok in seq.split()))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [count_shape_variety(s) for s in seqs]
        return sum((wi if a == b else 0) for wi, a, b in zip(w, y_true, y_pred)) / (
            sum(w) + 1e-9
        )


# ---------- synthetic fallback ----------------------------------
def make_synthetic_dataset(n):
    shapes = list(string.ascii_uppercase[:6])
    cols = list(string.ascii_lowercase[:6])
    seqs, labels = [], []
    for _ in range(n):
        tokens = [
            random.choice(shapes) + random.choice(cols)
            for _ in range(random.randint(4, 9))
        ]
        seqs.append(" ".join(tokens))
        labels.append(random.randint(0, 3))
    return {"sequence": seqs, "label": labels}


# ---------- load data -------------------------------------------
if HAVE_SPR:
    try:
        root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
        dsets = load_spr_bench(root)
        train_seqs, train_labels = dsets["train"]["sequence"], dsets["train"]["label"]
        dev_seqs, dev_labels = dsets["dev"]["sequence"], dsets["dev"]["label"]
        test_seqs, test_labels = dsets["test"]["sequence"], dsets["test"]["label"]
        print("Loaded SPR_BENCH from", root)
    except Exception as e:
        warnings.warn(f"Cannot load SPR_BENCH: {e}")
        HAVE_SPR = False
if not HAVE_SPR:
    train = make_synthetic_dataset(600)
    dev = make_synthetic_dataset(150)
    test = make_synthetic_dataset(300)
    train_seqs, train_labels = train["sequence"], train["label"]
    dev_seqs, dev_labels = dev["sequence"], dev["label"]
    test_seqs, test_labels = test["sequence"], test["label"]
    print("Using synthetic fallback dataset.")

n_classes = int(max(train_labels + dev_labels + test_labels)) + 1
print("Num classes:", n_classes)

# ---------- vocabulary ------------------------------------------
shape_letters = sorted({tok[0] for seq in train_seqs for tok in seq.split()})
color_letters = sorted(
    {tok[1] for seq in train_seqs for tok in seq.split() if len(tok) > 1}
)
shape2idx = {s: i + 1 for i, s in enumerate(shape_letters)}  # +1 reserve 0 for PAD
color2idx = {c: i + 1 for i, c in enumerate(color_letters)}
pad_s, pad_c = 0, 0


# ---------- dataset ---------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        shapes = [shape2idx.get(tok[0], 0) for tok in seq.split()]
        colors = [
            color2idx.get(tok[1], 0) if len(tok) > 1 else 0 for tok in seq.split()
        ]
        return {
            "shape_ids": torch.tensor(shapes, dtype=torch.long),
            "color_ids": torch.tensor(colors, dtype=torch.long),
            "s_var": torch.tensor(count_shape_variety(seq), dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq": seq,
        }


def collate(batch):
    lens = [len(b["shape_ids"]) for b in batch]
    max_len = max(lens)
    shape_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    color_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        shape_ids[i, : lens[i]] = b["shape_ids"]
        color_ids[i, : lens[i]] = b["color_ids"]
    s_var = torch.stack([b["s_var"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    seqs = [b["seq"] for b in batch]
    return {
        "shape_ids": shape_ids,
        "color_ids": color_ids,
        "lens": torch.tensor(lens),
        "s_var": s_var,
        "label": labels,
        "seqs": seqs,
    }


batch_size = 128
train_loader = DataLoader(
    SPRDataset(train_seqs, train_labels),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRDataset(dev_seqs, dev_labels),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRDataset(test_seqs, test_labels),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# ---------- model -----------------------------------------------
class NeuroSymGRU(nn.Module):
    def __init__(self, n_shape, n_color, n_class, d_s=16, d_c=16, h=64):
        super().__init__()
        self.embed_s = nn.Embedding(n_shape, d_s, padding_idx=pad_s)
        self.embed_c = nn.Embedding(n_color, d_c, padding_idx=pad_c)
        self.gru = nn.GRU(d_s + d_c, h, batch_first=True)
        self.fc = nn.Linear(h + 1, n_class)  # +1 for normalized shape variety

    def forward(self, shape_ids, color_ids, lens, s_var):  # lens on cpu ok
        emb = torch.cat([self.embed_s(shape_ids), self.embed_c(color_ids)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_n = h_n.squeeze(0)  # (B,h)
        s_feat = s_var.unsqueeze(1) / 10.0  # simple normalization
        return self.fc(torch.cat([h_n, s_feat], dim=1))


model = NeuroSymGRU(len(shape2idx) + 1, len(color2idx) + 1, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- experiment data container ---------------------------
experiment_data = {
    "spr_bench": {
        "epochs": [],
        "losses": {"train": [], "dev": []},
        "metrics": {"dev_SWA": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- training loop with early stopping -------------------
best_dev_swa, patience, wait, best_state = -1.0, 5, 0, None
max_epochs = 30
for epoch in range(1, max_epochs + 1):
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch["shape_ids"], batch["color_ids"], batch["lens"], batch["s_var"]
        )
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * batch["label"].size(0)
    tr_loss /= len(train_loader.dataset)

    # ---- validation
    model.eval()
    dev_loss, preds, gts, seq_buf = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(
                batch["shape_ids"], batch["color_ids"], batch["lens"], batch["s_var"]
            )
            dev_loss += criterion(out, batch["label"]).item() * batch["label"].size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            seq_buf.extend(batch["seqs"])
    dev_loss /= len(dev_loader.dataset)
    dev_swa = shape_weighted_accuracy(seq_buf, gts, preds)

    # logging
    experiment_data["spr_bench"]["epochs"].append(epoch)
    experiment_data["spr_bench"]["losses"]["train"].append(tr_loss)
    experiment_data["spr_bench"]["losses"]["dev"].append(dev_loss)
    experiment_data["spr_bench"]["metrics"]["dev_SWA"].append(dev_swa)
    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} dev_loss={dev_loss:.4f} dev_SWA={dev_swa:.4f}"
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

# ---------- restore best ----------------------------------------
if best_state:
    model.load_state_dict(best_state)

# ---------- test evaluation -------------------------------------
model.eval()
preds, gts, seq_buf = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(
            batch["shape_ids"], batch["color_ids"], batch["lens"], batch["s_var"]
        )
        preds.extend(out.argmax(1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
        seq_buf.extend(batch["seqs"])
test_swa = shape_weighted_accuracy(seq_buf, gts, preds)
print(f"\nTEST Shape-Weighted Accuracy = {test_swa:.4f}")

experiment_data["spr_bench"]["predictions"] = preds
experiment_data["spr_bench"]["ground_truth"] = gts
experiment_data["spr_bench"]["test_SWA"] = test_swa

# ---------- save artefacts --------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in ./working")
