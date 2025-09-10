import os, random, pathlib, gc, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict

# -------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------
# Helper to load SPR_BENCH or fall back to synthetic tiny set
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ("train", "dev", "test"):
        d[s] = _load(f"{s}.csv")
    return d


data_roots = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr = None
for p in data_roots:
    if (p / "train.csv").exists():
        spr = load_spr_bench(p)
        print("Loaded real SPR")
        break
if spr is None:
    print("SPR_BENCH not found, creating toy synthetic set")
    random.seed(0)

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labels = [], []
        for i in range(n):
            L = random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            seqs.append(seq)
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = DatasetDict()
    for sp, n in [("train", 800), ("dev", 200), ("test", 200)]:
        spr[sp] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )

# -------- vocabulary for shapes and colors --------------------
shape_vocab = {"<PAD>": 0}
color_vocab = {"<PAD>": 0}
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        s, c = tok[0], tok[1] if len(tok) > 1 else "0"
        if s not in shape_vocab:
            shape_vocab[s] = len(shape_vocab)
        if c not in color_vocab:
            color_vocab[c] = len(color_vocab)
shape_pad, color_pad = shape_vocab["<PAD>"], color_vocab["<PAD>"]
print(f"Shapes:{len(shape_vocab)}  Colors:{len(color_vocab)}")


# --------------- tokenisation -----------------
def tokenize(seq):
    return seq.strip().split()


def encode_token(token):
    s_idx = shape_vocab.get(token[0], shape_pad)
    c_idx = color_vocab.get(token[1] if len(token) > 1 else "0", color_pad)
    return s_idx, c_idx


# --------------- Dataset classes --------------
class ContrastiveSPRDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def random_mask(self, toks, p=0.15):
        out = []
        for t in toks:
            if random.random() < p:
                out.append("<MASK>")
            else:
                out.append(t)
        return out

    def local_shuffle(self, toks, window=3):
        toks = toks[:]
        i = 0
        while i < len(toks):
            w_len = random.randint(1, window)
            segment = toks[i : i + w_len]
            random.shuffle(segment)
            toks[i : i + w_len] = segment
            i += w_len
        return toks

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        view1 = self.random_mask(toks)
        view2 = self.local_shuffle(toks)
        return view1, view2


class ClassifierSPRDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


# ----------------- collate functions -----------------
def collate_views(batch):
    v1, v2 = zip(*batch)

    def encode(tok_lists):
        shp_ids = [
            torch.tensor([encode_token(t)[0] for t in toks], dtype=torch.long)
            for toks in tok_lists
        ]
        col_ids = [
            torch.tensor([encode_token(t)[1] for t in toks], dtype=torch.long)
            for toks in tok_lists
        ]
        lens = [len(x) for x in shp_ids]
        shp_ids = pad_sequence(shp_ids, batch_first=True, padding_value=shape_pad)
        col_ids = pad_sequence(col_ids, batch_first=True, padding_value=color_pad)
        return (
            shp_ids.to(device),
            col_ids.to(device),
            torch.tensor(lens, dtype=torch.long).to(device),
        )

    s1, c1, l1 = encode(v1)
    s2, c2, l2 = encode(v2)
    return {"s1": s1, "c1": c1, "l1": l1, "s2": s2, "c2": c2, "l2": l2}


def collate_class(batch):
    toks, labels, raw = zip(*batch)
    shp = [
        torch.tensor([encode_token(t)[0] for t in tok], dtype=torch.long)
        for tok in toks
    ]
    col = [
        torch.tensor([encode_token(t)[1] for t in tok], dtype=torch.long)
        for tok in toks
    ]
    lens = [len(x) for x in shp]
    shp = pad_sequence(shp, batch_first=True, padding_value=shape_pad).to(device)
    col = pad_sequence(col, batch_first=True, padding_value=color_pad).to(device)
    return {
        "shape": shp,
        "color": col,
        "len": torch.tensor(lens, dtype=torch.long).to(device),
        "label": torch.tensor(labels, dtype=torch.long).to(device),
        "seq": raw,
    }


# ----------------- Metrics ------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if a == b else 0 for wi, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if a == b else 0 for wi, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    corr = [wi if a == b else 0 for wi, a, b in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ------------------ Model -------------------
class SPRTransformerEncoder(nn.Module):
    def __init__(self, shape_vocab, color_vocab, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.shape_emb = nn.Embedding(len(shape_vocab), d_model)
        self.color_emb = nn.Embedding(len(color_vocab), d_model)
        self.pos_emb = nn.Embedding(128, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.cls_proj = nn.Linear(d_model, d_model)

    def forward(self, shp_ids, col_ids, lens):
        pos_idx = torch.arange(shp_ids.size(1), device=shp_ids.device).unsqueeze(0)
        x = self.shape_emb(shp_ids) + self.color_emb(col_ids) + self.pos_emb(pos_idx)
        mask = shp_ids == shape_pad
        h = self.encoder(x, src_key_padding_mask=mask)
        cls = h.mean(1)  # simple pooling
        return self.cls_proj(cls)


class SPRClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.cls_proj.out_features, num_classes)

    def forward(self, shp_ids, col_ids, lens):
        z = self.encoder(shp_ids, col_ids, lens)
        return self.head(z)


# ------------------ loss --------------------
def nt_xent(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    logits_12 = sim[:N, N:]
    logits_21 = sim[N:, :N]
    loss = (
        -torch.diag(logits_12)
        + torch.logsumexp(sim[:N], dim=1)
        - torch.diag(logits_21)
        + torch.logsumexp(sim[N:], dim=1)
    ).mean() * 0.5
    return loss


# ----------- Dataloaders -------------------
batch_c = 128
pretrain_loader = DataLoader(
    ContrastiveSPRDataset(spr["train"]),
    batch_size=batch_c,
    shuffle=True,
    collate_fn=collate_views,
)
train_loader = DataLoader(
    ClassifierSPRDataset(spr["train"]),
    batch_size=batch_c,
    shuffle=True,
    collate_fn=collate_class,
)
dev_loader = DataLoader(
    ClassifierSPRDataset(spr["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_class,
)

# -------------- Experiment record ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- Build model --------------
encoder = SPRTransformerEncoder(shape_vocab, color_vocab).to(device)
classifier = SPRClassifier(encoder, num_classes=len(set(spr["train"]["label"]))).to(
    device
)

# -------- Pre-training ---------------------
optim_pt = torch.optim.Adam(encoder.parameters(), lr=3e-4)
pretrain_epochs = 4
for ep in range(1, pretrain_epochs + 1):
    encoder.train()
    running = 0.0
    for b in pretrain_loader:
        optim_pt.zero_grad()
        z1 = encoder(b["s1"], b["c1"], b["l1"])
        z2 = encoder(b["s2"], b["c2"], b["l2"])
        loss = nt_xent(z1, z2)
        loss.backward()
        optim_pt.step()
        running += loss.item() * b["s1"].size(0)
    epoch_loss = running / len(pretrain_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(epoch_loss)
    print(f"Pretrain epoch {ep}/{pretrain_epochs}: loss={epoch_loss:.4f}")

# ---------- Fine-tuning --------------------
optim_ft = torch.optim.Adam(classifier.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
ft_epochs = 5
for ep in range(1, ft_epochs + 1):
    # train
    classifier.train()
    run_loss = 0.0
    for b in train_loader:
        optim_ft.zero_grad()
        logits = classifier(b["shape"], b["color"], b["len"])
        loss = criterion(logits, b["label"])
        loss.backward()
        optim_ft.step()
        run_loss += loss.item() * b["shape"].size(0)
    train_loss = run_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # val
    classifier.eval()
    vloss = 0.0
    preds = []
    gts = []
    seqs = []
    with torch.no_grad():
        for b in dev_loader:
            logits = classifier(b["shape"], b["color"], b["len"])
            loss = criterion(logits, b["label"])
            vloss += loss.item() * b["shape"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(b["label"].cpu().tolist())
            seqs.extend(b["seq"])
    val_loss = vloss / len(dev_loader.dataset)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    comp = complexity_weighted_accuracy(seqs, gts, preds)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"SWA": swa, "CWA": cwa, "CompWA": comp}
    )
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CompWA={comp:.3f}"
    )

    # simple early record of best preds
    if ep == ft_epochs:
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = gts

# -------------- Save metrics -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to ./working")

# cleanup
torch.cuda.empty_cache()
gc.collect()
