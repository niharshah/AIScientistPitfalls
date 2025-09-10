import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ----------------- experiment dict -----------------
experiment_data = {"learning_rate": {}}

# ----------------- reproducibility -----------------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- helper metrics -----------------
def count_shape_variety(sequence):
    return len({tok[0] for tok in sequence.split()})


def count_color_variety(sequence):
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa + 1e-12)


# ----------------- data -----------------
def load_or_create_dataset():
    root = pathlib.Path("SPR_BENCH")
    if root.exists():
        from SPR import load_spr_bench

        return load_spr_bench(root)

    def gen_row(_id):
        length = random.randint(4, 9)
        seq = " ".join(
            random.choice("ABCD") + random.choice("abcd") for _ in range(length)
        )
        return {"id": _id, "sequence": seq, "label": length % 2}

    return DatasetDict(
        {
            "train": HFDataset.from_list([gen_row(i) for i in range(500)]),
            "dev": HFDataset.from_list([gen_row(1_000 + i) for i in range(200)]),
            "test": HFDataset.from_list([gen_row(2_000 + i) for i in range(200)]),
        }
    )


spr = load_or_create_dataset()
print({k: len(v) for k, v in spr.items()})

# ----------------- vocab -----------------
all_text = " ".join(spr["train"]["sequence"])
vocab = sorted(set(all_text.split()))
tok2idx = {tok: i + 2 for i, tok in enumerate(vocab)}
tok2idx["<PAD>"] = 0
tok2idx["<UNK>"] = 1
idx2tok = {i: t for t, i in tok2idx.items()}


def encode(seq):
    return [tok2idx.get(tok, 1) for tok in seq.split()]


vocab_size = len(tok2idx)
print("Vocab size:", vocab_size)


# ----------------- dataset wrappers -----------------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lens = [len(b["x"]) for b in batch]
    mx = max(lens)
    xs = torch.zeros(len(batch), mx, dtype=torch.long)
    for i, b in enumerate(batch):
        xs[i, : lens[i]] = b["x"]
    ys = torch.stack([b["y"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": xs.to(device),
        "len": torch.tensor(lens).to(device),
        "y": ys.to(device),
        "raw_seq": raw,
    }


batch_size = 64
train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size, shuffle=False, collate_fn=collate
)
n_classes = len(set(spr["train"]["label"]))
print("Classes:", n_classes)


# ----------------- model -----------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid_dim=128, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x, lens):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ----------------- hyper-parameter grid -----------------
lr_grid = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
epochs = 5

for lr in lr_grid:
    tag = str(lr)
    experiment_data["learning_rate"][tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
    model = LSTMClassifier(vocab_size, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            optim.zero_grad()
            logits = model(batch["x"], batch["len"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optim.step()
            tr_loss += loss.item() * batch["y"].size(0)
        tr_loss /= len(train_loader.dataset)
        # ---- val ----
        model.eval()
        val_loss = 0.0
        all_seq, all_true, all_pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["x"], batch["len"])
                loss = criterion(logits, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                preds = torch.argmax(logits, 1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw_seq"])
        val_loss /= len(dev_loader.dataset)
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        # ---- log ----
        rec = experiment_data["learning_rate"][tag]["SPR_BENCH"]
        rec["losses"]["train"].append(tr_loss)
        rec["losses"]["val"].append(val_loss)
        rec["metrics"]["val"].append(hwa)
        rec["predictions"].append(all_pred)
        rec["ground_truth"].append(all_true)
        rec["timestamps"].append(time.time())
        print(
            f"[lr={lr}] Epoch {epoch}: train={tr_loss:.4f} val={val_loss:.4f} HWA={hwa:.3f}"
        )

# ----------------- save -----------------
os.makedirs("working", exist_ok=True)
np.save("working/experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
