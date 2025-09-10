import os, random, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ---------------------------------------------------------------------
# WORK DIR & GPU HANDLING ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# EXPERIMENT LOG --------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ---------------------------------------------------------------------
# BASIC METRICS ---------------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-9)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-9)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-9)


# ---------------------------------------------------------------------
# DATA LOADING ----------------------------------------------------------
def load_or_create_dataset():
    root = pathlib.Path("SPR_BENCH")
    if root.exists():
        from SPR import load_spr_bench

        return load_spr_bench(root)

    # tiny synthetic fallback
    def gen_row(_id):
        ln = random.randint(4, 10)
        shapes, colors = "ABCD", "abcd"
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(ln))
        return {"id": _id, "sequence": seq, "label": int(ln % 2)}

    train = [gen_row(i) for i in range(600)]
    dev = [gen_row(2000 + i) for i in range(200)]
    test = [gen_row(4000 + i) for i in range(200)]
    return DatasetDict(
        {
            "train": HFDataset.from_list(train),
            "dev": HFDataset.from_list(dev),
            "test": HFDataset.from_list(test),
        }
    )


spr = load_or_create_dataset()
print({k: len(v) for k, v in spr.items()})

# ---------------------------------------------------------------------
# VOCABULARY ------------------------------------------------------------
all_tokens = sorted(set(" ".join(spr["train"]["sequence"]).split()))
tok2idx = {tok: i + 4 for i, tok in enumerate(all_tokens)}  # +4 to leave specials
tok2idx["<PAD>"] = 0
tok2idx["<UNK>"] = 1
tok2idx["<MASK>"] = 2
tok2idx["<CLS>"] = 3
idx2tok = {i: t for t, i in tok2idx.items()}
vocab_size = len(tok2idx)
print(f"Vocab size = {vocab_size}")


def encode(seq):
    return [tok2idx.get(tok, tok2idx["<UNK>"]) for tok in seq.split()]


# ---------------------------------------------------------------------
# AUGMENTATIONS FOR CONTRASTIVE PRETRAIN -------------------------------
def mask_tokens(tokens, p=0.15):
    return [tok2idx["<MASK>"] if random.random() < p else tok for tok in tokens]


def shuffle_tokens(tokens, k=3):
    if len(tokens) <= 1:
        return tokens
    tokens = tokens.copy()
    for _ in range(k):
        i, j = random.sample(range(len(tokens)), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens


def make_view(token_ids):
    view = mask_tokens(token_ids, p=0.20)
    if random.random() < 0.5:
        view = shuffle_tokens(view, k=min(3, len(view)))
    return view


# ---------------------------------------------------------------------
# DATASETS --------------------------------------------------------------
class SPRContrastive(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        toks = encode(seq)
        v1 = make_view(toks)
        v2 = make_view(toks)
        return torch.tensor(v1, dtype=torch.long), torch.tensor(v2, dtype=torch.long)


class SPRClassification(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return encode(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def pad_batch(seqs):
    lens = [len(s) for s in seqs]
    mx = max(lens)
    out = torch.zeros(len(seqs), mx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out, torch.tensor(lens, dtype=torch.long)


def collate_contrastive(batch):
    v1, v2 = zip(*batch)
    x1, l1 = pad_batch(v1)
    x2, l2 = pad_batch(v2)
    return {
        "x1": x1.to(device),
        "l1": l1.to(device),
        "x2": x2.to(device),
        "l2": l2.to(device),
    }


def collate_classification(batch):
    seqs, labels, raw = zip(*batch)
    x, l = pad_batch(seqs)
    return {
        "x": x.to(device),
        "l": l.to(device),
        "y": torch.tensor(labels, dtype=torch.long).to(device),
        "raw_seq": raw,
    }


# ---------------------------------------------------------------------
# MODEL -----------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=tok2idx["<PAD>"])
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid * 2, 128)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # mean pooling
        mask = (x != tok2idx["<PAD>"]).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lens = mask.sum(1)
        mean = summed / torch.clamp(lens, min=1)
        return nn.functional.relu(self.proj(mean))


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x, lengths):
        z = self.encoder(x, lengths)
        return self.fc(z)


# ---------------------------------------------------------------------
# NT-XENT LOSS ----------------------------------------------------------
def nt_xent(z1, z2, temperature=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    pos_idx = torch.arange(B, device=z.device)
    positives = sim[pos_idx, pos_idx + B]
    positives = torch.cat([positives, sim[pos_idx + B, pos_idx]])
    denom = torch.logsumexp(sim, dim=1)
    loss = -positives + denom
    return loss.mean()


# ---------------------------------------------------------------------
# CONTRASTIVE PRETRAIN --------------------------------------------------
pretrain_loader = DataLoader(
    SPRContrastive(spr["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
    drop_last=True,
)

encoder = Encoder(vocab_size).to(device)
opt_enc = torch.optim.Adam(encoder.parameters(), lr=3e-4)

pre_epochs = 3
print("\n***** CONTRASTIVE PRE-TRAIN *****")
for ep in range(1, pre_epochs + 1):
    encoder.train()
    epoch_loss = 0.0
    for batch in pretrain_loader:
        opt_enc.zero_grad()
        z1 = encoder(batch["x1"], batch["l1"])
        z2 = encoder(batch["x2"], batch["l2"])
        loss = nt_xent(z1, z2)
        loss.backward()
        opt_enc.step()
        epoch_loss += loss.item()
    print(
        f"Pre-train Epoch {ep}/{pre_epochs} - loss {epoch_loss/len(pretrain_loader):.4f}"
    )

# ---------------------------------------------------------------------
# CLASSIFICATION FINE-TUNE ---------------------------------------------
train_loader = DataLoader(
    SPRClassification(spr["train"]),
    batch_size=64,
    shuffle=True,
    collate_fn=collate_classification,
)
val_loader = DataLoader(
    SPRClassification(spr["dev"]),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_classification,
)
n_classes = len(set(spr["train"]["label"]))
model = Classifier(encoder, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

fine_epochs = 10
print("\n***** SUPERVISED FINE-TUNE *****")
for ep in range(1, fine_epochs + 1):
    # ---- train ----
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["x"], batch["l"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * batch["y"].size(0)
    tr_loss /= len(train_loader.dataset)

    # ---- validate ----
    model.eval()
    va_loss, y_true, y_pred, raw_seq = 0.0, [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["x"], batch["l"])
            loss = criterion(logits, batch["y"])
            va_loss += loss.item() * batch["y"].size(0)
            preds = torch.argmax(logits, 1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch["y"].cpu().tolist())
            raw_seq.extend(batch["raw_seq"])
    va_loss /= len(val_loader.dataset)
    swa = shape_weighted_accuracy(raw_seq, y_true, y_pred)
    cwa = color_weighted_accuracy(raw_seq, y_true, y_pred)
    cowa = complexity_weighted_accuracy(raw_seq, y_true, y_pred)

    # ---- logging ----
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(va_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"SWA": swa, "CWA": cwa, "CoWA": cowa}
    )
    experiment_data["SPR_BENCH"]["predictions"].append(y_pred)
    experiment_data["SPR_BENCH"]["ground_truth"].append(y_true)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {ep}: validation_loss = {va_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CoWA={cowa:.3f}"
    )

# ---------------------------------------------------------------------
# SAVE ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to working directory")
