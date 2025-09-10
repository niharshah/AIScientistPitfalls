import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ----------------- experiment dict -----------------
experiment_data = {
    "baseline": {"SPR_BENCH": {}},
    "token_order_randomization": {"SPR_BENCH": {}},
}

# ----------------- device & seeds -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
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

    # tiny synthetic fallback
    def gen_row(_id):
        length = random.randint(4, 9)
        shapes, colors = "ABCD", "abcd"
        seq = " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )
        return {"id": _id, "sequence": seq, "label": int(length % 2)}

    train_rows = [gen_row(i) for i in range(600)]
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
tok2idx = {tok: i + 2 for i, tok in enumerate(vocab)}  # 0 PAD, 1 UNK
tok2idx["<PAD>"], tok2idx["<UNK>"] = 0, 1
idx2tok = {i: t for t, i in tok2idx.items()}
vocab_size = len(tok2idx)
print("Vocab size:", vocab_size)


def encode_sequence(seq):
    return [tok2idx.get(tok, 1) for tok in seq.strip().split()]


# ----------------- torch dataset -----------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_split, shuffle_internal=False):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]
        self.shuffle_internal = shuffle_internal

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        toks = seq.strip().split()
        if self.shuffle_internal:
            random.shuffle(toks)  # token-order randomization
        enc = torch.tensor([tok2idx.get(t, 1) for t in toks], dtype=torch.long)
        return {
            "x": enc,
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": seq,
        }  # keep original for metrics


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
train_loader_base = DataLoader(
    SPRTorchSet(spr["train"], shuffle_internal=False),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
train_loader_rand = DataLoader(
    SPRTorchSet(spr["train"], shuffle_internal=True),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"], shuffle_internal=False),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
n_classes = len(set(spr["train"]["label"]))
print("Num classes:", n_classes)


# ----------------- model -----------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, n_out=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_out)

    def forward(self, x, lengths):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ----------------- train / evaluate -----------------
def run_training(train_loader, num_epochs):
    model = LSTMClassifier(vocab_size, n_out=n_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = {
        "losses": {"train": [], "val": []},
        "metrics": {"val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, num_epochs + 1):
        model.train()
        tloss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch["x"], batch["len"])
            loss = crit(out, batch["y"])
            loss.backward()
            opt.step()
            tloss += loss.item() * batch["y"].size(0)
        train_loss = tloss / len(train_loader.dataset)

        # validation
        model.eval()
        vloss = 0.0
        all_seq = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(batch["x"], batch["len"])
                loss = crit(out, batch["y"])
                vloss += loss.item() * batch["y"].size(0)
                preds = torch.argmax(out, 1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw_seq"])
        val_loss = vloss / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(all_seq, y_true, y_pred)
        cwa = color_weighted_accuracy(all_seq, y_true, y_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)

        rec["losses"]["train"].append(train_loss)
        rec["losses"]["val"].append(val_loss)
        rec["metrics"]["val"].append(hwa)
        rec["predictions"].append(y_pred)
        rec["ground_truth"].append(y_true)
        rec["timestamps"].append(time.time())
        print(
            f"Ep{ep}/{num_epochs} | tr={train_loss:.3f} val={val_loss:.3f} HWA={hwa:.3f}"
        )
    return rec


# ----------------- sweep -----------------
epoch_options = [5, 10, 20, 30]
for e in epoch_options:
    print(f"\n=== Baseline, {e} epochs ===")
    experiment_data["baseline"]["SPR_BENCH"][str(e)] = run_training(
        train_loader_base, e
    )
    print(f"\n=== Token-Order Randomization, {e} epochs ===")
    experiment_data["token_order_randomization"]["SPR_BENCH"][str(e)] = run_training(
        train_loader_rand, e
    )

# ----------------- save -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
