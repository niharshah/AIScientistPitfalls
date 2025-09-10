import os, pathlib, random, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------------  GPU / Device handling  ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------  Dataset loading  ----------------------------------
def load_official():
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    d = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH from", DATA_PATH)
    return {k: d[k] for k in ("train", "dev", "test")}


def make_synthetic():
    print("Generating synthetic SPR toy dataset …")
    shapes, colours = ["A", "B", "C", "D"], ["r", "g", "b"]

    def synth(n):
        seqs, labels = [], []
        for i in range(n):
            length = random.randint(4, 12)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            lbl = int(
                sum(t[0] == "A" for t in seq.split()) % 2 == 0
            )  # arbitrary parity rule
            seqs.append(seq)
            labels.append(lbl)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    return {"train": synth(3000), "dev": synth(600), "test": synth(600)}


try:
    raw = load_official()
except Exception as e:
    print("Fallback reason:", e)
    raw = make_synthetic()


# helper for SWA if official SPR present else simple acc
def _count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [_count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ---------------------  Vocabulary & Symbolic feats -----------------------
tok2id = {"<PAD>": 0, "<UNK>": 1}
for seq in raw["train"]["sequence"]:
    for tok in seq.split():
        if tok not in tok2id:
            tok2id[tok] = len(tok2id)
pad_id, unk_id = tok2id["<PAD>"], tok2id["<UNK>"]

shape_set, colour_set = set(), set()
for seq in raw["train"]["sequence"]:
    for tok in seq.split():
        shape_set.add(tok[0])
        if len(tok) > 1:
            colour_set.add(tok[1])
shape2id = {s: i for i, s in enumerate(sorted(shape_set))}
colour2id = {c: i for i, c in enumerate(sorted(colour_set))}
sym_dim = len(shape2id) + len(colour2id) + 1  # +1 for length feature


def encode_tokens(seq):
    return [tok2id.get(t, unk_id) for t in seq.split()]


def symbolic_vec(seq):
    vec = np.zeros(sym_dim, dtype=np.float32)
    for tok in seq.split():
        vec[shape2id[tok[0]]] += 1
        if len(tok) > 1:
            vec[len(shape2id) + colour2id[tok[1]]] += 1
    vec[-1] = len(seq.split())  # sequence length feature
    return vec


# ---------------------  Torch Dataset -------------------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.ids, self.seqs, self.labels = (
            split["id"],
            split["sequence"],
            split["label"],
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        return {
            "seq_ids": torch.tensor(encode_tokens(s), dtype=torch.long),
            "sym": torch.tensor(symbolic_vec(s)),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": s,
        }


def collate(batch):
    lens = [len(b["seq_ids"]) for b in batch]
    maxlen = max(lens)
    seq = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        seq[i, : lens[i]] = b["seq_ids"]
    labels = torch.stack([b["label"] for b in batch])
    sym = torch.stack([b["sym"] for b in batch])
    rawseq = [b["raw"] for b in batch]
    return {
        "seq": seq,
        "lengths": torch.tensor(lens),
        "sym": sym,
        "label": labels,
        "raw": rawseq,
    }


train_ds, val_ds, test_ds = (
    SPRDataset(raw["train"]),
    SPRDataset(raw["dev"]),
    SPRDataset(raw["test"]),
)


# ---------------------  Model ---------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # x: B,L,E
        return x + self.pe[:, : x.size(1)]


class NeuroSymbolicTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, sym_dim, n_classes, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.sym_mlp = nn.Sequential(
            nn.Linear(sym_dim, 64), nn.ReLU(), nn.Linear(64, d_model)
        )
        self.gate = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, seq, lengths, sym):
        mask = seq == pad_id
        h = self.embed(seq)
        h = self.pos(h)
        h = self.transformer(h, src_key_padding_mask=mask)
        # mean pool over valid positions
        lengths = lengths.unsqueeze(1)
        sum_h = torch.sum(h.masked_fill(mask.unsqueeze(2), 0.0), dim=1)
        neu_feat = sum_h / lengths
        sym_feat = self.sym_mlp(sym)
        gate = torch.sigmoid(self.gate(torch.cat([neu_feat, sym_feat], dim=-1)))
        fused = gate * neu_feat + (1 - gate) * sym_feat
        return self.classifier(fused)


# ---------------------  Training utils ------------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss, preds, trues, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["seq"], batch["lengths"], batch["sym"])
            loss = criterion(out, batch["label"])
            tot_loss += loss.item() * len(batch["label"])
            p = out.argmax(-1).cpu().tolist()
            preds.extend(p)
            trues.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw"])
    swa = shape_weighted_accuracy(seqs, trues, preds)
    return tot_loss / len(trues), swa, preds, trues


# ---------------------  Hyper-params & DataLoaders ------------------------
BS, EPOCHS = 64, 6
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

num_classes = len(set(raw["train"]["label"]))
model = NeuroSymbolicTransformer(
    len(tok2id),
    d_model=64,
    nhead=4,
    nlayers=2,
    sym_dim=sym_dim,
    n_classes=num_classes,
    pad=pad_id,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------  Training loop -------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    run_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["seq"], batch["lengths"], batch["sym"])
        loss = criterion(out, batch["label"])
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * len(batch["label"])
    train_loss = run_loss / len(train_ds)
    val_loss, val_swa, _, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.4f}")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)

# ---------------------  Final Test ----------------------------------------
test_loss, test_swa, preds, gts = evaluate(model, test_loader)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_swa

# ---------------------  Save results --------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
