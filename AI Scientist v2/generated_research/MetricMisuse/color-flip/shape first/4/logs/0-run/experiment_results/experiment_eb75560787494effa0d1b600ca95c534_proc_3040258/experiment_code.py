import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ----------------------------- experiment dict -----------------------------
experiment_data = {"baseline": {}, "factorized": {}}

# ----------------------------- device & seeds ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
print("Using device:", device)


# ----------------------------- helper metrics ------------------------------
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


# ----------------------------- data loading --------------------------------
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

# ----------------------------- vocabularies --------------------------------
all_text = " ".join(spr["train"]["sequence"])
tok_vocab = sorted(set(all_text.split()))
tok2idx = {tok: i + 2 for i, tok in enumerate(tok_vocab)}  # 0 PAD, 1 UNK
tok2idx["<PAD>"], tok2idx["<UNK>"] = 0, 1
idx2tok = {i: t for t, i in tok2idx.items()}
vocab_size = len(tok2idx)
print("Token vocab size:", vocab_size)

# shape & color vocab
shape_vocab = sorted({tok[0] for tok in tok_vocab})
color_vocab = sorted({tok[1] for tok in tok_vocab})
shape2idx = {s: i + 2 for i, s in enumerate(shape_vocab)}
color2idx = {c: i + 2 for i, c in enumerate(color_vocab)}
shape2idx["<PAD>"], shape2idx["<UNK>"] = 0, 1
color2idx["<PAD>"], color2idx["<UNK>"] = 0, 1
shape_vocab_size, color_vocab_size = len(shape2idx), len(color2idx)
print("Shape vocab:", shape_vocab_size, "Color vocab:", color_vocab_size)


# ----------------------------- encoders ------------------------------------
def encode_token(tok):
    return tok2idx.get(tok, 1)


def encode_shapes(seq):
    res = []
    for tok in seq.strip().split():
        res.append(shape2idx.get(tok[0], 1) if tok else 1)
    return res


def encode_colors(seq):
    res = []
    for tok in seq.strip().split():
        res.append(color2idx.get(tok[1], 1) if len(tok) > 1 else 1)
    return res


# ----------------------------- torch dataset ------------------------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "token": torch.tensor(
                [encode_token(t) for t in seq.split()], dtype=torch.long
            ),
            "shape": torch.tensor(encode_shapes(seq), dtype=torch.long),
            "color": torch.tensor(encode_colors(seq), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": seq,
        }


def pad_batch(col, pad_val=0):
    lengths = [len(x) for x in col]
    maxlen = max(lengths)
    out = torch.full((len(col), maxlen), pad_val, dtype=torch.long)
    for i, x in enumerate(col):
        out[i, : len(x)] = x
    return out, torch.tensor(lengths, dtype=torch.long)


def collate(batch):
    token_col = [b["token"] for b in batch]
    shape_col = [b["shape"] for b in batch]
    color_col = [b["color"] for b in batch]
    tok_pad, lens = pad_batch(token_col, 0)
    shp_pad, _ = pad_batch(shape_col, 0)
    col_pad, _ = pad_batch(color_col, 0)
    ys = torch.stack([b["y"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "token": tok_pad.to(device),
        "shape": shp_pad.to(device),
        "color": col_pad.to(device),
        "len": lens.to(device),
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


# ----------------------------- models --------------------------------------
class BaselineLSTM(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, n_out=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_out)

    def forward(self, token_ids, lengths):
        em = self.emb(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


class FactorizedLSTM(nn.Module):
    def __init__(self, shape_vocab, color_vocab, emb=64, hid=128, n_out=2):
        super().__init__()
        self.shape_emb = nn.Embedding(shape_vocab, emb, padding_idx=0)
        self.color_emb = nn.Embedding(color_vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_out)

    def forward(self, shape_ids, color_ids, lengths):
        em = self.shape_emb(shape_ids) + self.color_emb(color_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


# ----------------------------- training loop ------------------------------
def run_training(model_type: str, num_epochs: int):
    if model_type == "baseline":
        model = BaselineLSTM(vocab_size, n_out=n_classes).to(device)
    else:
        model = FactorizedLSTM(shape_vocab_size, color_vocab_size, n_out=n_classes).to(
            device
        )

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            if model_type == "baseline":
                out = model(batch["token"], batch["len"])
            else:
                out = model(batch["shape"], batch["color"], batch["len"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optim.step()
            tloss += loss.item() * batch["y"].size(0)
        tr_loss = tloss / len(train_loader.dataset)

        # validation
        model.eval()
        vloss, all_seq, y_true, y_pred = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                if model_type == "baseline":
                    out = model(batch["token"], batch["len"])
                else:
                    out = model(batch["shape"], batch["color"], batch["len"])
                loss = criterion(out, batch["y"])
                vloss += loss.item() * batch["y"].size(0)
                preds = torch.argmax(out, 1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw_seq"])
        val_loss = vloss / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(all_seq, y_true, y_pred)
        cwa = color_weighted_accuracy(all_seq, y_true, y_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)

        rec["losses"]["train"].append(tr_loss)
        rec["losses"]["val"].append(val_loss)
        rec["metrics"]["val"].append(hwa)
        rec["predictions"].append(y_pred)
        rec["ground_truth"].append(y_true)
        rec["timestamps"].append(time.time())
        print(
            f"[{model_type}] Ep {ep}/{num_epochs}  tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  HWA={hwa:.3f}"
        )
    return rec


# ----------------------------- hyper-parameter sweep -----------------------
epoch_options = [5, 10]
for variant in ["baseline", "factorized"]:
    experiment_data[variant]["SPR_BENCH"] = {}
    for e in epoch_options:
        print(f"\n=== {variant.upper()} | training for {e} epochs ===")
        experiment_data[variant]["SPR_BENCH"][str(e)] = run_training(variant, e)

# ----------------------------- save ----------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
