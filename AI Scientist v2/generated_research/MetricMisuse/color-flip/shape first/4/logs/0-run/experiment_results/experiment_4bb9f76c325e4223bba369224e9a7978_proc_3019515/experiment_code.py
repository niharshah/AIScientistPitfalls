import os, random, pathlib, time, numpy as np, torch, gc
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ---------- experiment dict ----------
experiment_data = {"num_epochs_tuning": {"SPR_BENCH": {}}}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- metrics ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa + 1e-12)


# ---------- data ----------
def load_or_create_dataset():
    root = pathlib.Path("SPR_BENCH")
    if root.exists():
        from SPR import load_spr_bench

        return load_spr_bench(root)

    def gen_row(_id):
        length = random.randint(4, 9)
        shapes, colors = "ABCD", "abcd"
        seq = " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )
        label = int(length % 2)
        return {"id": _id, "sequence": seq, "label": label}

    train = [gen_row(i) for i in range(500)]
    dev = [gen_row(1_000 + i) for i in range(200)]
    test = [gen_row(2_000 + i) for i in range(200)]
    return DatasetDict(
        {
            "train": HFDataset.from_list(train),
            "dev": HFDataset.from_list(dev),
            "test": HFDataset.from_list(test),
        }
    )


spr = load_or_create_dataset()
print({k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
all_text = " ".join(spr["train"]["sequence"])
vocab = sorted(set(all_text.split()))
tok2idx = {tok: i + 2 for i, tok in enumerate(vocab)}
tok2idx["<PAD>"] = 0
tok2idx["<UNK>"] = 1
idx2tok = {i: t for t, i in tok2idx.items()}
vocab_size = len(tok2idx)
print("Vocab size:", vocab_size)


def encode_sequence(seq):
    return [tok2idx.get(tok, 1) for tok in seq.strip().split()]


# ---------- dataset & loader ----------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    xs = [b["x"] for b in batch]
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    maxlen = lens.max().item()
    pad = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        pad[i, : len(x)] = x
    ys = torch.stack([b["y"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": pad.to(device),
        "len": lens.to(device),
        "y": ys.to(device),
        "raw_seq": raw,
    }


batch_size = 64
train_loader = lambda: DataLoader(
    SPRTorchSet(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
n_classes = len(set(spr["train"]["label"]))
print("Num classes:", n_classes)


# ---------- model ----------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_classes)

    def forward(self, x, lengths):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ---------- training routine ----------
def run_for_epochs(num_epochs: int):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = LSTMClassifier(vocab_size, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses, hwas = [], [], []
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader():
            optim.zero_grad()
            logits = model(batch["x"], batch["len"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * batch["y"].size(0)
        train_loss = epoch_loss / len(spr["train"])
        train_losses.append(train_loss)

        model.eval()
        vloss, seqs, ys, preds = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["x"], batch["len"])
                loss = criterion(logits, batch["y"])
                vloss += loss.item() * batch["y"].size(0)
                p = torch.argmax(logits, 1).cpu().tolist()
                preds.extend(p)
                ys.extend(batch["y"].cpu().tolist())
                seqs.extend(batch["raw_seq"])
        vloss /= len(spr["dev"])
        val_losses.append(vloss)
        swa = shape_weighted_accuracy(seqs, ys, preds)
        cwa = color_weighted_accuracy(seqs, ys, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        hwas.append(hwa)
        print(
            f"[{num_epochs}ep] Epoch {epoch}/{num_epochs} | train {train_loss:.4f} | val {vloss:.4f} | HWA {hwa:.3f}"
        )

    return {
        "losses": {"train": train_losses, "val": val_losses},
        "metrics": {"val": hwas},
        "predictions": preds,
        "ground_truth": ys,
        "timestamps": time.time(),
    }


# ---------- hyperparameter sweep ----------
for ep in [5, 10, 20, 30]:
    print(f"\n=== Training model for {ep} epochs ===")
    result = run_for_epochs(ep)
    experiment_data["num_epochs_tuning"]["SPR_BENCH"][str(ep)] = result
    # clear GPU
    torch.cuda.empty_cache()
    gc.collect()

# ---------- save ----------
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
