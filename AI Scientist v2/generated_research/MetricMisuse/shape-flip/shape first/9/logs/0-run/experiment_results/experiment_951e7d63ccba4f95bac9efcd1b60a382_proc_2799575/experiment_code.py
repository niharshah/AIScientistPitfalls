import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader

# ---------- working dir & experiment data ----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------- device ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- load SPR benchmark or synthetic fall-back ----------------------
try:
    from SPR import (
        load_spr_bench,
        count_shape_variety,
        count_color_variety,
        shape_weighted_accuracy,
    )

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    spr_ds = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH from", DATA_PATH)
except Exception as e:
    print("Could not load SPR_BENCH – using tiny synthetic data instead:", e)
    shapes, colours = ["A", "B", "C"], ["r", "g", "b"]

    def synth(n):
        seqs, labels = [], []
        for i in range(n):
            length = random.randint(4, 8)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            # arbitrary rule: label 1 if more than 1 unique shape
            labels.append(int(len(set(tok[0] for tok in seq.split())) > 1))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_ds = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def count_shape_variety(seq):
        return len(set(tok[0] for tok in seq.split()))

    def count_color_variety(seq):
        return len(set(tok[1] for tok in seq.split()))

    def shape_weighted_accuracy(seqs, y_t, y_p):
        w = [count_shape_variety(s) for s in seqs]
        return sum((yt == yp) * wt for yt, yp, wt in zip(y_t, y_p, w)) / max(sum(w), 1)


# ---------- vocabulary -----------------------------------------------------
all_train_tokens = [tok for seq in spr_ds["train"]["sequence"] for tok in seq.split()]
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok in set(all_train_tokens):
    vocab[tok] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]


def encode_seq(seq: str):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


# ---------- dataset / dataloader ------------------------------------------
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
        seq = self.seqs[idx]
        return {
            "seq_ids": torch.tensor(encode_seq(seq), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "sym_feats": torch.tensor(
                [count_shape_variety(seq), count_color_variety(seq), len(seq.split())],
                dtype=torch.float32,
            ),
            "raw_seq": seq,
        }


def collate_fn(batch):
    lens = [len(x["seq_ids"]) for x in batch]
    max_len = max(lens)
    seq_tensor = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        seq_tensor[i, : lens[i]] = b["seq_ids"]
    labels = torch.stack([b["label"] for b in batch])
    sym = torch.stack([b["sym_feats"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "seq": seq_tensor,
        "lengths": torch.tensor(lens),
        "sym": sym,
        "label": labels,
        "raw_seq": raw,
    }


train_ds = SPRDataset(spr_ds["train"])
val_ds = SPRDataset(spr_ds["dev"])
test_ds = SPRDataset(spr_ds["test"])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

num_classes = len(set(spr_ds["train"]["label"]))


# ---------- Neural-symbolic model -----------------------------------------
class NeuroSymbolic(nn.Module):
    def __init__(self, vocab_sz, embed_dim, hidden_dim, classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.neural_head = nn.Linear(hidden_dim * 2, classes)

        self.symbolic_head = nn.Linear(3, classes)  # 3 symbolic feats
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2 + 3, 1), nn.Sigmoid())

    def forward(self, seq, lengths, sym_feats):
        emb = self.embed(seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # [B,2H]
        neural_logits = self.neural_head(h)  # [B,C]
        sym_logits = self.symbolic_head(sym_feats)  # [B,C]
        # gate between 0 and 1
        g = self.gate(torch.cat([h, sym_feats], dim=-1))  # [B,1]
        logits = g * sym_logits + (1 - g) * neural_logits
        return logits, neural_logits, sym_logits, g.squeeze(-1)


# ---------- training utilities --------------------------------------------
criterion = nn.CrossEntropyLoss()


def run_eval(model, loader):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    seqs = []
    with torch.no_grad():
        for batch in loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits, _, _, _ = model(b["seq"], b["lengths"], b["sym"])
            loss = criterion(logits, b["label"])
            total_loss += loss.item() * b["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(b["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    avg_loss = total_loss / len(labels)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    return avg_loss, swa, preds, labels, seqs


# ---------- training loop --------------------------------------------------
model = NeuroSymbolic(len(vocab), 32, 64, num_classes, pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 8

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    for batch in train_loader:
        b = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits, _, _, _ = model(b["seq"], b["lengths"], b["sym"])
        loss = criterion(logits, b["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * b["label"].size(0)
    train_loss = running_loss / len(train_ds)
    val_loss, val_swa, *_ = run_eval(model, val_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.4f}")

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

# ---------- final test evaluation -----------------------------------------
test_loss, test_swa, preds, labels, seqs = run_eval(model, test_loader)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_swa
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels

# ---------- save experiment data ------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
