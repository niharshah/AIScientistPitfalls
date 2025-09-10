import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ----------------- experiment dict -----------------
experiment_data = {
    "embedding_dim_tuning": {"SPR_BENCH": {}}  # will be filled with emb_XX sub-dicts
}


# ----------------- reproducibility -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- helper metrics -----------------
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


# ----------------- data loading -----------------
def load_or_create_dataset():
    root = pathlib.Path("SPR_BENCH")
    if root.exists():
        from SPR import load_spr_bench

        return load_spr_bench(root)

    # fallback synthetic tiny data
    def gen_row(_id):
        length = random.randint(4, 9)
        shapes = "ABCD"
        colors = "abcd"
        seq = " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )
        label = int(length % 2)  # dummy rule
        return {"id": _id, "sequence": seq, "label": label}

    train_rows = [gen_row(i) for i in range(500)]
    dev_rows = [gen_row(1000 + i) for i in range(200)]
    test_rows = [gen_row(2000 + i) for i in range(200)]
    return DatasetDict(
        {
            "train": HFDataset.from_list(train_rows),
            "dev": HFDataset.from_list(dev_rows),
            "test": HFDataset.from_list(test_rows),
        }
    )


spr = load_or_create_dataset()
print({k: len(v) for k, v in spr.items()})

# ----------------- vocabulary -----------------
all_text = " ".join(spr["train"]["sequence"])
vocab = sorted(list(set(all_text.split())))
tok2idx = {tok: i + 2 for i, tok in enumerate(vocab)}
tok2idx["<PAD>"] = 0
tok2idx["<UNK>"] = 1
idx2tok = {i: t for t, i in tok2idx.items()}
vocab_size = len(tok2idx)
print(f"Vocab size: {vocab_size}")


def encode_sequence(seq):
    return [tok2idx.get(tok, 1) for tok in seq.strip().split()]


# ----------------- torch dataset -----------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

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
    lens = [len(x) for x in xs]
    maxlen = max(lens)
    xs_pad = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        xs_pad[i, : len(x)] = x
    ys = torch.stack([b["y"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": xs_pad.to(device),
        "len": torch.tensor(lens, dtype=torch.long).to(device),
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
print(f"Num classes: {n_classes}")


# ----------------- model -----------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid_dim=128, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x, lengths):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ----------------- hyperparameter sweep -----------------
embed_dims = [32, 64, 128, 256]
best_hwa, best_dim = -1.0, None

for emb_dim in embed_dims:
    print(f"\n--- Training with embedding_dim={emb_dim} ---")
    # sub-dict for this dim
    dim_key = f"emb_{emb_dim}"
    experiment_data["embedding_dim_tuning"]["SPR_BENCH"][dim_key] = {
        "metrics": {"val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    model = LSTMClassifier(vocab_size, emb_dim=emb_dim, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optim.zero_grad()
            logits = model(batch["x"], batch["len"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optim.step()
            train_loss_sum += loss.item() * batch["y"].size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        # ---- eval ----
        model.eval()
        val_loss_sum, all_seq, all_true, all_pred = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["x"], batch["len"])
                loss = criterion(logits, batch["y"])
                val_loss_sum += loss.item() * batch["y"].size(0)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw_seq"])
        val_loss = val_loss_sum / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)

        # ---- logging ----
        edict = experiment_data["embedding_dim_tuning"]["SPR_BENCH"][dim_key]
        edict["losses"]["train"].append(train_loss)
        edict["losses"]["val"].append(val_loss)
        edict["metrics"]["val"].append(hwa)
        edict["predictions"].append(all_pred)
        edict["ground_truth"].append(all_true)
        edict["timestamps"].append(time.time())

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )

    # ---- best tracking ----
    final_hwa = edict["metrics"]["val"][-1]
    if final_hwa > best_hwa:
        best_hwa, best_dim = final_hwa, emb_dim

print(f"\nBest embedding_dim={best_dim} with HWA={best_hwa:.3f}")

# ----------------- save experiment -----------------
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
