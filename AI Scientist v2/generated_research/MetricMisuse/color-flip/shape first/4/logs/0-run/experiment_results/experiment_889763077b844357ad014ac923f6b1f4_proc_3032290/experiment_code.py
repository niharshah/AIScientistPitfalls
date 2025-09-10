import os, random, pathlib, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import DatasetDict, Dataset as HFDataset

# ---------- dirs ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data ----------
experiment_data = {
    "SPR_Contrastive": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ---------- metric helpers ----------
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


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


# ---------- data loading ----------
def load_or_create_dataset():
    try:
        from SPR import load_spr_bench

        root = pathlib.Path("SPR_BENCH")
        if root.exists():
            return load_spr_bench(root)
        raise FileNotFoundError
    except Exception:
        # tiny synthetic fallback
        def gen_row(_id):
            L = random.randint(4, 9)
            shapes, colors = "ABCD", "abcd"
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            return {"id": _id, "sequence": seq, "label": int(L % 2)}

        train = [gen_row(i) for i in range(800)]
        dev = [gen_row(2000 + i) for i in range(200)]
        test = [gen_row(4000 + i) for i in range(200)]
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
tok2idx = {tok: i + 4 for i, tok in enumerate(vocab)}  # extra specials
specials = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "<CLS>": 3}
tok2idx.update(specials)
idx2tok = {i: t for t, i in tok2idx.items()}
vocab_size = len(tok2idx)


def encode(seq):
    return [tok2idx.get(tok, 1) for tok in seq.strip().split()]


# ---------- augmentation ----------
def augment_tokens(tokens, mask_prob=0.15, shuffle_prob=0.2):
    toks = tokens[:]
    # shuffle a small window with probability
    if random.random() < shuffle_prob and len(toks) > 3:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    # mask
    toks = [2 if random.random() < mask_prob else t for t in toks]
    return toks


# ---------- datasets ----------
class PretrainSet(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = encode(self.seqs[idx])
        view1 = augment_tokens(tokens)
        view2 = augment_tokens(tokens)
        return torch.tensor(view1), torch.tensor(view2)


class FineTuneSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(encode(self.seqs[idx])),
            torch.tensor(self.labels[idx]),
            self.seqs[idx],
        )


def pad_collate_pretrain(batch):
    v1, v2 = zip(*batch)
    lengths = [len(x) for x in v1]
    maxlen = max(lengths)

    def pad(tensors):
        out = torch.zeros(len(tensors), maxlen, dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, : len(t)] = t
        return out

    return {
        "v1": pad(v1).to(device),
        "v2": pad(v2).to(device),
        "len": torch.tensor(lengths).to(device),
    }


def pad_collate_finetune(batch):
    xs, ys, raws = zip(*batch)
    lengths = [len(x) for x in xs]
    maxlen = max(lengths)
    out = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, t in enumerate(xs):
        out[i, : len(t)] = t
    return {
        "x": out.to(device),
        "len": torch.tensor(lengths).to(device),
        "y": torch.stack(ys).to(device),
        "raw": list(raws),
    }


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=64, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=8, dim_feedforward=hid * 2, batch_first=True
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_param = nn.Parameter(torch.randn(emb))

    def forward(self, x, lengths):
        # prepend CLS token embedding (parameter) for pooling
        B = x.size(0)
        cls = self.cls_param.unsqueeze(0).repeat(B, 1).unsqueeze(1)  # (B,1,E)
        emb = self.emb(x)  # (B,L,E)
        inp = torch.cat([cls, emb], dim=1)
        enc = self.tr(inp)
        return enc[:, 0]  # CLS position


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(self.enc.cls_param.numel(), num_classes)

    def forward(self, x, lengths):
        z = self.enc(x, lengths)
        return self.fc(z), z


# ---------- contrastive loss ----------
def nt_xent(h1, h2, tau=0.5):
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    logits = torch.mm(h1, h2.t()) / tau
    labels = torch.arange(h1.size(0), device=device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return loss / 2


# ---------- pre-training ----------
def contrastive_pretrain(encoder, epochs=2, batch_size=128):
    loader = DataLoader(
        PretrainSet(spr["train"]["sequence"]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_pretrain,
    )
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.train()
    for ep in range(1, epochs + 1):
        total = 0
        n = 0
        for batch in loader:
            opt.zero_grad()
            z1 = encoder(batch["v1"], batch["len"])
            z2 = encoder(batch["v2"], batch["len"])
            loss = nt_xent(z1, z2)
            loss.backward()
            opt.step()
            total += loss.item() * batch["v1"].size(0)
            n += batch["v1"].size(0)
        print(f"[Pretrain] Epoch {ep} contrastive_loss = {total/n:.4f}")


# ---------- fine-tune ----------
def fine_tune(encoder, epochs=5, batch_size=64):
    train_loader = DataLoader(
        FineTuneSet(spr["train"]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_finetune,
    )
    val_loader = DataLoader(
        FineTuneSet(spr["dev"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_finetune,
    )
    num_classes = len(set(spr["train"]["label"]))
    model = Classifier(encoder, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        # train
        model.train()
        t_loss = 0
        for batch in train_loader:
            opt.zero_grad()
            out, _ = model(batch["x"], batch["len"])
            loss = criterion(out, batch["y"])
            loss.backward()
            opt.step()
            t_loss += loss.item() * batch["y"].size(0)
        train_loss = t_loss / len(train_loader.dataset)
        # val
        model.eval()
        v_loss = 0
        seqs = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in val_loader:
                out, _ = model(batch["x"], batch["len"])
                loss = criterion(out, batch["y"])
                v_loss += loss.item() * batch["y"].size(0)
                pred = out.argmax(1).cpu().tolist()
                y_pred.extend(pred)
                y_true.extend(batch["y"].cpu().tolist())
                seqs.extend(batch["raw"])
        val_loss = v_loss / len(val_loader.dataset)
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        cowa = complexity_weighted_accuracy(seqs, y_true, y_pred)
        print(
            f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CoWA={cowa:.3f}"
        )
        # record
        experiment_data["SPR_Contrastive"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_Contrastive"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_Contrastive"]["metrics"]["train"].append({"CoWA": None})
        experiment_data["SPR_Contrastive"]["metrics"]["val"].append({"CoWA": cowa})
        experiment_data["SPR_Contrastive"]["predictions"].append(y_pred)
        experiment_data["SPR_Contrastive"]["ground_truth"].append(y_true)
        experiment_data["SPR_Contrastive"]["timestamps"].append(time.time())
    return model


# ---------- run ----------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
enc = Encoder(vocab_size).to(device)
contrastive_pretrain(enc, epochs=2)
model = fine_tune(enc, epochs=5)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
