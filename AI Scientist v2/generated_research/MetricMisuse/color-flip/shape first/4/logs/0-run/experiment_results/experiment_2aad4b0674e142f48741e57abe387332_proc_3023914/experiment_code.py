import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ---------- experiment log container ----------
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}

# ---------- working dir ----------
work_dir = os.path.join(os.getcwd(), "working")
os.makedirs(work_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- metrics ----------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
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

    return DatasetDict(
        {
            "train": HFDataset.from_list([gen_row(i) for i in range(500)]),
            "dev": HFDataset.from_list([gen_row(1000 + i) for i in range(200)]),
            "test": HFDataset.from_list([gen_row(2000 + i) for i in range(200)]),
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


def encode_sequence(seq):
    return [tok2idx.get(tok, 1) for tok in seq.strip().split()]


print("Vocab size:", vocab_size)


# ---------- torch datasets ----------
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
    lengths = [len(x) for x in xs]
    maxlen = max(lengths)
    pad = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        pad[i, : len(x)] = x
    ys = torch.stack([b["y"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": pad.to(device),
        "len": torch.tensor(lengths).to(device),
        "y": ys.to(device),
        "raw_seq": raw,
    }


batch_size = 64
train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)

n_classes = len(set(spr["train"]["label"]))
print("Num classes:", n_classes)


# ---------- model def ----------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid_dim=128, n_cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, n_cls)

    def forward(self, x, l):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, l.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ---------- training routine ----------
def run_trial(lr, epochs=5):
    print(f"\n=== LR {lr:.1e} ===")
    model = LSTMClassifier(vocab_size, n_cls=n_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch["x"], batch["len"])
            loss = crit(logits, batch["y"])
            loss.backward()
            opt.step()
            loss_sum += loss.item() * batch["y"].size(0)
            total += batch["y"].size(0)
        train_loss = loss_sum / total
        # -------- eval ---------
        model.eval()
        val_loss = 0.0
        seqs = []
        y_t = []
        y_p = []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["x"], batch["len"])
                loss = crit(logits, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                preds = torch.argmax(logits, 1).cpu().tolist()
                y_p.extend(preds)
                y_t.extend(batch["y"].cpu().tolist())
                seqs.extend(batch["raw_seq"])
        val_loss /= len(dev_loader.dataset)
        swa = shape_weighted_accuracy(seqs, y_t, y_p)
        cwa = color_weighted_accuracy(seqs, y_t, y_p)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        history["losses"]["train"].append(train_loss)
        history["losses"]["val"].append(val_loss)
        history["metrics"]["val"].append(hwa)
        history["predictions"].append(y_p)
        history["ground_truth"].append(y_t)
        history["timestamps"].append(time.time())
        print(
            f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} HWA={hwa:.3f}"
        )
    return history


# ---------- hyperparameter sweep ----------
lr_grid = [1e-4, 3e-4, 1e-3, 3e-3]
for lr in lr_grid:
    history = run_trial(lr)
    experiment_data["learning_rate"]["SPR_BENCH"][str(lr)] = history

# ---------- save ----------
np.save(os.path.join(work_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
