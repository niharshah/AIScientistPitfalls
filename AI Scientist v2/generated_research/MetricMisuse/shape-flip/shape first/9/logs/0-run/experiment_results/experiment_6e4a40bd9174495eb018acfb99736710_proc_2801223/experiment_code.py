import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

# -----------------------  Experiment store --------------------------------
experiment_data = {
    "BinarySymFeat": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# -----------------------  Device ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------  Data loading ------------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    raw = load_spr_bench(DATA_PATH)
    print("Loaded official SPR_BENCH.")
except Exception as e:
    print("SPR_BENCH not found – generating synthetic toy data.", e)
    shapes, colours = ["A", "B", "C", "D"], ["r", "g", "b"]

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            labels.append(int(any(tok[0] == "A" for tok in seq.split())))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    raw = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def shape_weighted_accuracy(seqs, y_t, y_p):
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)


# -----------------------  Vocabularies ------------------------------------
tok_counter = {}
for seq in raw["train"]["sequence"]:
    for tok in seq.split():
        tok_counter[tok] = tok_counter.get(tok, 0) + 1

tok2id = {"<PAD>": 0, "<UNK>": 1}
for tok in tok_counter:
    tok2id[tok] = len(tok2id)
pad_id, unk_id = tok2id["<PAD>"], tok2id["<UNK>"]

shape2id = {}
for seq in raw["train"]["sequence"]:
    for tok in seq.split():
        s = tok[0]
        if s not in shape2id:
            shape2id[s] = len(shape2id)
shape_feat_dim = len(shape2id)
num_classes = len(set(raw["train"]["label"]))


# -----------------------  Encoders ----------------------------------------
def encode_tokens(seq):
    return [tok2id.get(t, unk_id) for t in seq.split()]


def encode_shape_presence(seq):
    vec = np.zeros(shape_feat_dim, dtype=np.float32)
    for tok in set(seq.split()):  # set so only presence matters
        s = tok[0]
        if s in shape2id:
            vec[shape2id[s]] = 1.0
    return vec


# -----------------------  Dataset / Dataloader ----------------------------
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
        seq_str = self.seqs[idx]
        return {
            "seq_ids": torch.tensor(encode_tokens(seq_str), dtype=torch.long),
            "shape_bin": torch.tensor(encode_shape_presence(seq_str)),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": seq_str,
        }


def collate(batch):
    lengths = [len(b["seq_ids"]) for b in batch]
    maxlen = max(lengths)
    seqs = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        seqs[i, : lengths[i]] = b["seq_ids"]
    labels = torch.stack([b["label"] for b in batch])
    shp = torch.stack([b["shape_bin"] for b in batch])
    rawseq = [b["raw_seq"] for b in batch]
    return {
        "seq": seqs,
        "lengths": torch.tensor(lengths),
        "shape_bin": shp,
        "label": labels,
        "raw_seq": rawseq,
    }


train_ds, val_ds, test_ds = (
    SPRDataset(raw["train"]),
    SPRDataset(raw["dev"]),
    SPRDataset(raw["test"]),
)


# -----------------------  Model -------------------------------------------
class NeuroSymbolicClassifier(nn.Module):
    def __init__(self, vocab, embed_dim, hid_dim, shape_dim, n_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hid_dim, batch_first=True, bidirectional=True)
        self.neu_proj = nn.Linear(hid_dim * 2, 64)
        self.sym_proj = nn.Linear(shape_dim, 32)
        self.cls = nn.Linear(96, n_classes)

    def forward(self, seq, lengths, shape_bin):
        emb = self.embed(seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        neu = torch.relu(self.neu_proj(h))
        sym = torch.relu(self.sym_proj(shape_bin))
        return self.cls(torch.cat([neu, sym], -1))


# -----------------------  Utils -------------------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    L, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["seq"], batch["lengths"], batch["shape_bin"])
            L += criterion(out, batch["label"]).item() * len(batch["label"])
            p = out.argmax(-1).cpu().tolist()
            preds.extend(p)
            gts.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    return L / len(gts), shape_weighted_accuracy(seqs, gts, preds), preds, gts


# -----------------------  Training ----------------------------------------
BS, EPOCHS = 32, 6
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

model = NeuroSymbolicClassifier(
    len(tok2id), 64, 128, shape_feat_dim, num_classes, pad_id
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(1, EPOCHS + 1):
    model.train()
    t0, running = time.time(), 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt.zero_grad()
        out = model(batch["seq"], batch["lengths"], batch["shape_bin"])
        loss = criterion(out, batch["label"])
        loss.backward()
        opt.step()
        running += loss.item() * len(batch["label"])
    train_loss = running / len(train_ds)
    val_loss, val_swa, _, _ = evaluate(model, val_loader)
    print(
        f"Epoch {ep}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_SWA={val_swa:.4f} ({time.time()-t0:.1f}s)"
    )
    experiment_data["BinarySymFeat"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["BinarySymFeat"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["BinarySymFeat"]["SPR_BENCH"]["metrics"]["train"].append(None)
    experiment_data["BinarySymFeat"]["SPR_BENCH"]["metrics"]["val"].append(val_swa)

# -----------------------  Test --------------------------------------------
test_loss, test_swa, preds, gts = evaluate(model, test_loader)
print(f"Test SWA = {test_swa:.4f}")
exp = experiment_data["BinarySymFeat"]["SPR_BENCH"]
exp["predictions"], exp["ground_truth"] = preds, gts
exp["metrics"]["test"] = test_swa

# -----------------------  Save --------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
