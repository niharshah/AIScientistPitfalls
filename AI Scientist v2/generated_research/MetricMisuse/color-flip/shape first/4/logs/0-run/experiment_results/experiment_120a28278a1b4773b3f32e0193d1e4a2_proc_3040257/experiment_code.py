import os, random, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, Dataset as HFDataset

# ---------- reproducibility & device ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------- helper metrics ----------
def count_shape_variety(sequence: str):  # shape = first char of token
    return len({tok[0] for tok in sequence.split() if tok})


def count_color_variety(sequence: str):  # color = second char of token
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa + 1e-12)


# ---------- synthetic dataset generator ----------
def gen_dataset(
    name, shapes, colors, length_rng, label_fn, n_train=600, n_dev=200, n_test=200
):
    def make_seq():
        L = random.randint(*length_rng)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def build_split(n, start_id):
        rows = []
        for i in range(start_id, start_id + n):
            seq = make_seq()
            tok_list = seq.split()
            label = label_fn(tok_list)
            rows.append({"id": i, "sequence": seq, "label": int(label)})
        return HFDataset.from_list(rows)

    return name, DatasetDict(
        {
            "train": build_split(n_train, 0),
            "dev": build_split(n_dev, 10_000),
            "test": build_split(n_test, 20_000),
        }
    )


# label functions (binary)
def lbl_parity(tok_list):  # 1 if even length, else 0
    return len(tok_list) % 2 == 0


def lbl_shape_gt_color(tok_list):  # 1 if #unique shapes > #unique colors
    shapes = len({t[0] for t in tok_list})
    colors = len({t[1] for t in tok_list})
    return shapes > colors


def lbl_first_equals_last(tok_list):  # 1 if first & last shapes equal
    return tok_list[0][0] == tok_list[-1][0]


# ---------- create 3 synthetic datasets ----------
datasets = {}
cfgs = [
    dict(
        name="SetA",
        shapes="ABCD",
        colors="abcd",
        length_rng=(4, 9),
        label_fn=lbl_parity,
    ),
    dict(
        name="SetB",
        shapes="EFGH",
        colors="efghij",
        length_rng=(6, 12),
        label_fn=lbl_shape_gt_color,
    ),
    dict(
        name="SetC",
        shapes="IJK",
        colors="klmno",
        length_rng=(3, 7),
        label_fn=lbl_first_equals_last,
    ),
]
for c in cfgs:
    n, ds = gen_dataset(**c)
    datasets[n] = ds
print({k: {s: len(v[s]) for s in v} for k, v in datasets.items()})


# ---------- torch helpers ----------
class TorchSeqSet(Dataset):
    def __init__(self, hf_split, tok2idx):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.tok2idx = tok2idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = [self.tok2idx.get(tok, 1) for tok in seq.split()]  # 1 == UNK
        return {"x": torch.tensor(ids), "y": torch.tensor(self.labels[idx]), "raw": seq}


def build_vocab(train_sequences):
    vocab = sorted({tok for seq in train_sequences for tok in seq.split()})
    tok2idx = {tok: i + 2 for i, tok in enumerate(vocab)}
    tok2idx["<PAD>"] = 0
    tok2idx["<UNK>"] = 1
    return tok2idx


def collate_fn(batch):
    lens = [len(b["x"]) for b in batch]
    maxlen = max(lens)
    xpad = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        xpad[i, : lens[i]] = b["x"]
    y = torch.stack([b["y"] for b in batch])
    raw = [b["raw"] for b in batch]
    return {
        "x": xpad.to(device),
        "len": torch.tensor(lens).to(device),
        "y": y.to(device),
        "raw": raw,
    }


# ---------- model ----------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128, n_out=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_out)

    def forward(self, x, l):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, l.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ---------- experiment container ----------
experiment_data = {"MultiSynthetic": {}}
EPOCHS = 10
BATCH = 64
for train_name, ds_train in datasets.items():
    print(f"\n=== Training on {train_name} ===")
    # vocab & loaders
    tok2idx = build_vocab(ds_train["train"]["sequence"])
    train_loader = DataLoader(
        TorchSeqSet(ds_train["train"], tok2idx),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        TorchSeqSet(ds_train["dev"], tok2idx),
        batch_size=BATCH,
        shuffle=False,
        collate_fn=collate_fn,
    )
    n_classes = len(set(ds_train["train"]["label"]))
    model = LSTMClassifier(len(tok2idx), n_out=n_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {"train": [], "val": []},
        "ground_truth": {"train": [], "val": []},
        "cross_eval": {},
    }

    # ---- training loop ----
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = 0
        for b in train_loader:
            opt.zero_grad()
            out = model(b["x"], b["len"])
            loss = crit(out, b["y"])
            loss.backward()
            opt.step()
            tl += loss.item() * b["y"].size(0)
        tr_loss = tl / len(train_loader.dataset)
        rec["losses"]["train"].append(tr_loss)

        # validation (in-domain)
        model.eval()
        vl = 0
        y_true = []
        y_pred = []
        seqs = []
        with torch.no_grad():
            for b in dev_loader:
                out = model(b["x"], b["len"])
                loss = crit(out, b["y"])
                vl += loss.item() * b["y"].size(0)
                preds = torch.argmax(out, 1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(b["y"].cpu().tolist())
                seqs.extend(b["raw"])
        val_loss = vl / len(dev_loader.dataset)
        rec["losses"]["val"].append(val_loss)
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        rec["metrics"]["val"].append(hwa)
        rec["predictions"]["val"].append(y_pred)
        rec["ground_truth"]["val"].append(y_true)
        print(
            f"Ep{ep:02d}: tr_loss={tr_loss:.3f} val_loss={val_loss:.3f} HWA={hwa:.3f}"
        )

    # ---- cross-domain evaluation ----
    model.eval()
    for eval_name, ds_eval in datasets.items():
        eval_loader = DataLoader(
            TorchSeqSet(ds_eval["dev"], tok2idx),
            batch_size=BATCH,
            shuffle=False,
            collate_fn=collate_fn,
        )
        y_true = []
        y_pred = []
        seqs = []
        with torch.no_grad():
            for b in eval_loader:
                out = model(b["x"], b["len"])
                preds = torch.argmax(out, 1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(b["y"].cpu().tolist())
                seqs.extend(b["raw"])
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        rec["cross_eval"][eval_name] = {
            "accuracy": acc,
            "swa": swa,
            "cwa": cwa,
            "hwa": hwa,
            "y_pred": y_pred,
            "y_true": y_true,
        }
    experiment_data["MultiSynthetic"][train_name] = rec

# ---------- save ----------
work_dir = os.path.join(os.getcwd(), "working")
pathlib.Path(work_dir).mkdir(exist_ok=True)
np.save(os.path.join(work_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
