import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ----------------- experiment buffer -----------------
experiment_data = {
    "dropout_rate": {
        "SPR_BENCH": {
            "rates": [],
            "metrics": {"train": [], "val": []},  # harmonic WA per epoch
            "losses": {"train": [], "val": []},  # CE loss per epoch
            "predictions": [],  # last-epoch predictions
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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

    # synthetic fallback tiny data
    def gen_row(_id):
        length = random.randint(4, 9)
        seq = " ".join(
            random.choice("ABCD") + random.choice("abcd") for _ in range(length)
        )
        return {"id": _id, "sequence": seq, "label": int(length % 2)}

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
vocab = sorted(set(all_text.split()))
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


# ----------------- model with dropout -----------------
class LSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size, emb_dim=64, hid_dim=128, n_classes=2, dropout_rate=0.0
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x, lengths):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = self.drop(h[-1])
        return self.fc(h)


# ----------------- training routine -----------------
def train_one_setting(drop_rate, epochs=5, lr=1e-3):
    model = LSTMClassifier(vocab_size, n_classes=n_classes, dropout_rate=drop_rate).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    per_ep_train_loss, per_ep_val_loss, per_ep_hwa = [], [], []
    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()
        running = 0.0
        for batch in train_loader:
            optim.zero_grad()
            logits = model(batch["x"], batch["len"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optim.step()
            running += loss.item() * batch["y"].size(0)
        tr_loss = running / len(train_loader.dataset)
        # ---- evaluation ----
        model.eval()
        val_running = 0.0
        all_seq = []
        all_true = []
        all_pred = []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["x"], batch["len"])
                loss = criterion(logits, batch["y"])
                val_running += loss.item() * batch["y"].size(0)
                preds = torch.argmax(logits, 1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw_seq"])
        val_loss = val_running / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        per_ep_train_loss.append(tr_loss)
        per_ep_val_loss.append(val_loss)
        per_ep_hwa.append(hwa)
        print(
            f"[drop={drop_rate}] Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  HWA={hwa:.3f}"
        )
    # ---- store results ----
    ed = experiment_data["dropout_rate"]["SPR_BENCH"]
    ed["rates"].append(drop_rate)
    ed["losses"]["train"].append(per_ep_train_loss)
    ed["losses"]["val"].append(per_ep_val_loss)
    ed["metrics"]["val"].append(per_ep_hwa)
    ed["predictions"].append(all_pred)
    ed["ground_truth"].append(all_true)
    ed["timestamps"].append(time.time())


# ----------------- run hyper-parameter sweep -----------------
for rate in [0.0, 0.1, 0.3, 0.5]:
    train_one_setting(rate)

# ----------------- save experiment -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
