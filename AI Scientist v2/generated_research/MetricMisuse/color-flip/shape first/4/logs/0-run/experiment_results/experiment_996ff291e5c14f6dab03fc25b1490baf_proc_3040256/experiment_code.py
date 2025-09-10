import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ----------------- experiment dict (baseline + ablation) -----------------
experiment_data = {
    "baseline": {"SPR_BENCH": {}},
    "frozen_emb": {"SPR_BENCH": {}},
}

# ----------------- device & seeds -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
print("Using device:", device)


# ----------------- helper metrics -----------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa + 1e-12)


# ----------------- data -----------------
def load_or_create_dataset():
    root = pathlib.Path("SPR_BENCH")
    if root.exists():
        from SPR import load_spr_bench

        return load_spr_bench(root)

    def gen_row(_id):
        L = random.randint(4, 9)
        shapes, colors = "ABCD", "abcd"
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        return {"id": _id, "sequence": seq, "label": int(L % 2)}

    train = [gen_row(i) for i in range(600)]
    dev = [gen_row(1000 + i) for i in range(200)]
    test = [gen_row(2000 + i) for i in range(200)]
    return DatasetDict(
        {
            "train": HFDataset.from_list(train),
            "dev": HFDataset.from_list(dev),
            "test": HFDataset.from_list(test),
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
vocab_size = len(tok2idx)


def encode(seq):
    return [tok2idx.get(tok, 1) for tok in seq.split()]


print("Vocab size:", vocab_size)


# ----------------- torch dataset -----------------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

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
    maxlen = max(lens)
    x_pad = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        x_pad[i, : lens[i]] = b["x"]
    y = torch.stack([b["y"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": x_pad.to(device),
        "len": torch.tensor(lens).to(device),
        "y": y.to(device),
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


# ----------------- train / eval -----------------
def run_training(num_epochs, freeze_emb=False):
    model = LSTMClassifier(vocab_size, n_out=n_classes).to(device)
    if freeze_emb:
        model.emb.weight.requires_grad = False
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
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
            optim.zero_grad()
            out = model(batch["x"], batch["len"])
            loss = crit(out, batch["y"])
            loss.backward()
            optim.step()
            tloss += loss.item() * batch["y"].size(0)
        train_loss = tloss / len(train_loader.dataset)

        model.eval()
        vloss = 0.0
        seqs = []
        y_t = []
        y_p = []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(batch["x"], batch["len"])
                loss = crit(out, batch["y"])
                vloss += loss.item() * batch["y"].size(0)
                preds = torch.argmax(out, 1).cpu().tolist()
                y_p.extend(preds)
                y_t.extend(batch["y"].cpu().tolist())
                seqs.extend(batch["raw_seq"])
        val_loss = vloss / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(seqs, y_t, y_p)
        cwa = color_weighted_accuracy(seqs, y_t, y_p)
        hwa = harmonic_weighted_accuracy(swa, cwa)

        rec["losses"]["train"].append(train_loss)
        rec["losses"]["val"].append(val_loss)
        rec["metrics"]["val"].append(hwa)
        rec["predictions"].append(y_p)
        rec["ground_truth"].append(y_t)
        rec["timestamps"].append(time.time())
        tag = "Frozen" if freeze_emb else "Baseline"
        print(
            f"{tag} | Epoch {ep}/{num_epochs} - tr_loss:{train_loss:.4f} val_loss:{val_loss:.4f} HWA:{hwa:.3f}"
        )
    return rec


# ----------------- sweeps -----------------
epoch_options = [5, 10, 20, 30]
for e in epoch_options:
    # baseline
    experiment_data["baseline"]["SPR_BENCH"][str(e)] = run_training(e, freeze_emb=False)
    # frozen embedding ablation
    experiment_data["frozen_emb"]["SPR_BENCH"][str(e)] = run_training(
        e, freeze_emb=True
    )

# ----------------- save -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
