import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ---------- experiment dict ----------
experiment_data = {"hid_dim": {}}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------- helper metrics ----------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


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
        return {"id": _id, "sequence": seq, "label": int(length % 2)}

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


def encode(seq):
    return [tok2idx.get(tok, 1) for tok in seq.split()]


vocab_size = len(tok2idx)
print("Vocab:", vocab_size)


# ---------- torch dataset ----------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


def collate(b):
    xs = [e["x"] for e in b]
    lens = [len(x) for x in xs]
    maxlen = max(lens)
    pad = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        pad[i, : len(x)] = x
    ys = torch.stack([e["y"] for e in b])
    raws = [e["raw"] for e in b]
    return {
        "x": pad.to(device),
        "len": torch.tensor(lens).to(device),
        "y": ys.to(device),
        "raw": raws,
    }


batch_size = 64
train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
n_classes = len(set(spr["train"]["label"]))


# ---------- model ----------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_classes):
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


# ---------- training util ----------
def run_experiment(hid_dim, epochs=5):
    model = LSTMClassifier(vocab_size, 64, hid_dim, n_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logs = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch["x"], batch["len"])
            loss = crit(logits, batch["y"])
            loss.backward()
            opt.step()
            tl += loss.item() * batch["y"].size(0)
        train_loss = tl / len(train_loader.dataset)
        # eval
        model.eval()
        vl, seqs, ys, ps = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["x"], batch["len"])
                loss = crit(logits, batch["y"])
                vl += loss.item() * batch["y"].size(0)
                pred = torch.argmax(logits, 1).cpu().tolist()
                ps.extend(pred)
                ys.extend(batch["y"].cpu().tolist())
                seqs.extend(batch["raw"])
        val_loss = vl / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(seqs, ys, ps)
        cwa = color_weighted_accuracy(seqs, ys, ps)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        logs["losses"]["train"].append(train_loss)
        logs["losses"]["val"].append(val_loss)
        logs["metrics"]["val"].append(hwa)
        logs["predictions"].append(ps)
        logs["ground_truth"].append(ys)
        logs["timestamps"].append(time.time())
        print(
            f"H{hid_dim} Ep{ep}: train {train_loss:.3f} val {val_loss:.3f} HWA {hwa:.3f}"
        )
    return logs


# ---------- hyperparameter sweep ----------
for hd in [64, 128, 256, 512]:
    print(f"\n=== Training with hidden dim {hd} ===")
    experiment_data["hid_dim"][f"h{hd}"] = {"SPR_BENCH": run_experiment(hd)}

# ---------- save ----------
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
