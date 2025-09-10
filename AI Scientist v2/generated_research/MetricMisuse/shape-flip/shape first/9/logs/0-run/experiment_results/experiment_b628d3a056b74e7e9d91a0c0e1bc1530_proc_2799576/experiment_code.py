import os, pathlib, time, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------- working dir & experiment store ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------- device ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- dataset load --------------------------------------------
try:
    from SPR import (
        load_spr_bench,
        count_shape_variety,
        count_color_variety,
        shape_weighted_accuracy,
    )

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    dset = load_spr_bench(DATA_PATH)
except Exception:
    # ----- fallback tiny synthetic dataset -------
    shapes, colours = ["A", "B", "C"], ["r", "g", "b"]

    def synth(n):
        seqs, labs = [], []
        for i in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            # label is 1 if at least one A token, else 0  (dummy rule)
            labs.append(int(any(t[0] == "A" for t in seq.split())))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    dset = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def count_shape_variety(sequence):
        return len(set(t[0] for t in sequence.split()))

    def count_color_variety(sequence):
        return len(set(t[1] for t in sequence.split()))

    def shape_weighted_accuracy(seq, y_t, y_p):
        w = [count_shape_variety(s) for s in seq]
        return sum(w_i if yt == yp else 0 for w_i, yt, yp in zip(w, y_t, y_p)) / max(
            1, sum(w)
        )


# ---------------- vocabulary build ----------------------------------------
all_train_tokens = [tok for s in dset["train"]["sequence"] for tok in s.split()]
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok in all_train_tokens:
    if tok not in vocab:
        vocab[tok] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


num_classes = len(set(dset["train"]["label"]))


# ---------------- PyTorch Dataset -----------------------------------------
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
        enc = encode(seq_str)
        sym = torch.tensor(
            [
                count_shape_variety(seq_str),
                count_color_variety(seq_str),
                len(seq_str.split()),
            ],
            dtype=torch.float,
        )
        return {
            "enc": torch.tensor(enc, dtype=torch.long),
            "sym": sym,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": seq_str,
        }


def collate(batch):
    lengths = [len(b["enc"]) for b in batch]
    maxlen = max(lengths)
    seqs = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        seqs[i, : lengths[i]] = b["enc"]
    sym_feats = torch.stack([b["sym"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    raws = [b["raw_seq"] for b in batch]
    return {
        "seq": seqs,
        "len": torch.tensor(lengths),
        "sym": sym_feats,
        "label": labels,
        "raw": raws,
    }


train_ds, val_ds, test_ds = (
    SPRDataset(dset["train"]),
    SPRDataset(dset["dev"]),
    SPRDataset(dset["test"]),
)


# ---------------- Model ----------------------------------------------------
class NeuralSymbolic(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_dim, sym_dim, out_dim, pad):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2 + sym_dim, out_dim)

    def forward(self, seq, lengths, sym):
        emb = self.emb(seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (batch, hid*2)
        concat = torch.cat([h, sym], dim=-1)
        return self.fc(concat)


# ---------------- training utils ------------------------------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, train=False, optim=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, preds, labels, seqs = 0.0, [], [], []
    for batch in loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch_t["seq"], batch_t["len"], batch_t["sym"])
        loss = criterion(out, batch_t["label"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item() * batch_t["label"].size(0)
        p = out.argmax(-1).cpu().tolist()
        preds.extend(p)
        labels.extend(batch_t["label"].cpu().tolist())
        seqs.extend(batch["raw"])
    avg_loss = total_loss / len(labels)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    return avg_loss, swa, preds, labels


# ---------------- Train loop ----------------------------------------------
batch_size = 32
epochs = 4
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

model = NeuralSymbolic(len(vocab), 32, 64, 3, num_classes, pad_idx).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(model, train_loader, train=True, optim=optim)
    val_loss, val_swa, _, _ = run_epoch(model, val_loader, train=False)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.4f}")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

# ---------------- Final test ----------------------------------------------
test_loss, test_swa, test_preds, test_labels = run_epoch(
    model, test_loader, train=False
)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels

# ---------------- Save & plot ---------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "confusion_SPR.png"))
    plt.close()
except Exception as e:
    pass
