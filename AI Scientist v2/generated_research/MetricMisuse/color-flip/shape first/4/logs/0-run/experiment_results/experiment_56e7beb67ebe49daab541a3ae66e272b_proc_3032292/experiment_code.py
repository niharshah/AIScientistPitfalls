import os, random, time, pathlib, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ----------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "pretrain": {"loss": []},
        "finetune": {
            "loss_train": [],
            "loss_val": [],
            "SWA": [],
            "CWA": [],
            "CoWA": [],
            "timestamps": [],
        },
    }
}

# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------
# metric helpers
def count_shape(sequence):  # first char of each token
    return len({tok[0] for tok in sequence.split() if tok})


def count_color(sequence):  # second char
    return len({tok[1] for tok in sequence.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) or 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) or 1)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape(s) * count_color(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) or 1)


# ----------------------------------------------------------------------
# dataset loading (falls back to tiny synthetic if SPR_BENCH missing)
def load_spr():
    root = pathlib.Path("SPR_BENCH")
    if root.exists():
        from SPR import load_spr_bench

        return load_spr_bench(root)
    # synthetic fallback
    shapes, colors = "ABCD", "abcd"

    def gen_row(i):
        ln = random.randint(4, 9)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(ln))
        return {"id": i, "sequence": seq, "label": ln % 3}

    train = [gen_row(i) for i in range(800)]
    dev = [gen_row(1_000 + i) for i in range(200)]
    test = [gen_row(2_000 + i) for i in range(200)]
    return DatasetDict(
        {
            "train": HFDataset.from_list(train),
            "dev": HFDataset.from_list(dev),
            "test": HFDataset.from_list(test),
        }
    )


spr = load_spr()
print({k: len(v) for k, v in spr.items()})

# ----------------------------------------------------------------------
# vocabulary
all_tokens = set()
for seq in spr["train"]["sequence"]:
    all_tokens.update(seq.split())
tok2idx = {tok: i + 4 for i, tok in enumerate(sorted(all_tokens))}
specials = ["<PAD>", "<UNK>", "<MASK>", "<CLS>"]
for i, sp in enumerate(specials):
    tok2idx[sp] = i
idx2tok = {i: t for t, i in tok2idx.items()}
PAD, UNK, MASK, CLS = [tok2idx[t] for t in specials]
vocab_size = len(tok2idx)
print("Vocab size", vocab_size)


def encode(seq):
    return [tok2idx.get(t, UNK) for t in seq.split()]


# ----------------------------------------------------------------------
# augmentations for contrastive pretraining
def aug_tokens(tokens):
    tokens = tokens[:]  # copy
    # random masking
    for i in range(len(tokens)):
        if random.random() < 0.2:
            tokens[i] = MASK
    # local shuffle (swap adjacent with p)
    i = 0
    while i < len(tokens) - 1:
        if random.random() < 0.1:
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
            i += 2
        else:
            i += 1
    return tokens


# ----------------------------------------------------------------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_split, unlabeled=False):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"] if not unlabeled else [-1] * len(self.seqs)
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tok_ids = encode(self.seqs[idx])
        return {
            "input": torch.tensor(tok_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def pad_collate(batch):
    xs = [b["input"] for b in batch]
    lens = [len(x) for x in xs]
    maxlen = max(lens) + 1  # +1 for CLS
    padded = torch.full((len(xs), maxlen), PAD, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, 0] = CLS
        padded[i, 1 : 1 + len(x)] = x
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": padded.to(device),
        "len": torch.tensor(lens, device=device),
        "y": labels.to(device),
        "raw_seq": raw,
    }


# ----------------------------------------------------------------------
batch_size = 256
pretrain_loader = DataLoader(
    SPRTorchSet(spr["train"], unlabeled=True),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=pad_collate,
)

train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=pad_collate
)

dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=128, shuffle=False, collate_fn=pad_collate
)

n_classes = len(set(spr["train"]["label"]))
print("classes", n_classes)


# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Encoder(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.tr = nn.TransformerEncoder(encoder_layer, nlayers)
        self.d_model = d_model

    def forward(self, x):
        emb = self.emb(x) * math.sqrt(self.d_model)
        emb = self.pos(emb)
        mask = x == PAD
        h = self.tr(emb.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        # mean pool excluding PAD
        mask_float = (~mask).float()
        pooled = (h * mask_float.unsqueeze(-1)).sum(1) / mask_float.sum(1, keepdim=True)
        return pooled  # [B, d_model]


class SPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
        self.cls_head = nn.Linear(128, n_classes)

    def forward(self, x, mode="cls"):
        h = self.encoder(x)
        if mode == "proj":
            z = F.normalize(self.proj(h), dim=1)
            return z
        return self.cls_head(h)


model = SPRModel().to(device)

# ----------------------------------------------------------------------
# contrastive pre-training ------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
temperature = 0.5
epochs_pre = 3


def contrastive_step(batch_x):
    # create two augmented views
    bsz = batch_x.size(0)
    tokens = batch_x.cpu().tolist()
    view1, view2 = [], []
    for toks in tokens:
        toks = [t for t in toks if t != PAD]
        a1 = aug_tokens(toks[1:])  # skip CLS
        a2 = aug_tokens(toks[1:])
        for arr, target in ((view1, a1), (view2, a2)):
            pad_len = batch_x.size(1) - 1 - len(target)
            arr.append([CLS] + target + [PAD] * pad_len)
    v1 = torch.tensor(view1, dtype=torch.long, device=device)
    v2 = torch.tensor(view2, dtype=torch.long, device=device)

    z1 = model(v1, mode="proj")
    z2 = model(v2, mode="proj")
    z = torch.cat([z1, z2], 0)  # 2B x d
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / temperature
    labels = torch.arange(bsz, device=device)
    labels = torch.cat([labels + bsz, labels])
    mask = torch.eye(2 * bsz, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, -9e15)
    loss = F.cross_entropy(sim, labels)
    return loss


print("\n--- Pre-training ---")
for ep in range(1, epochs_pre + 1):
    model.train()
    epoch_loss = 0
    for batch in pretrain_loader:
        optimizer.zero_grad()
        loss = contrastive_step(batch["x"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["x"].size(0)
    avg_loss = epoch_loss / len(pretrain_loader.dataset)
    experiment_data["SPR_BENCH"]["pretrain"]["loss"].append(avg_loss)
    print(f"Contrastive Epoch {ep}: loss={avg_loss:.4f}")

# ----------------------------------------------------------------------
# fine-tuning -----------------------------------------------------------
ft_epochs = 8
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n--- Fine-tuning ---")
for ep in range(1, ft_epochs + 1):
    # ---- train
    model.train()
    tloss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        tloss += loss.item() * batch["y"].size(0)
    train_loss = tloss / len(train_loader.dataset)
    # ---- validation
    model.eval()
    vloss = 0
    all_seq = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dev_loader:
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            vloss += loss.item() * batch["y"].size(0)
            pred = torch.argmax(logits, 1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(batch["y"].cpu().tolist())
            all_seq.extend(batch["raw_seq"])
    val_loss = vloss / len(dev_loader.dataset)
    swa = shape_weighted_accuracy(all_seq, y_true, y_pred)
    cwa = color_weighted_accuracy(all_seq, y_true, y_pred)
    coa = complexity_weighted_accuracy(all_seq, y_true, y_pred)
    experiment_data["SPR_BENCH"]["finetune"]["loss_train"].append(train_loss)
    experiment_data["SPR_BENCH"]["finetune"]["loss_val"].append(val_loss)
    experiment_data["SPR_BENCH"]["finetune"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["finetune"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["finetune"]["CoWA"].append(coa)
    experiment_data["SPR_BENCH"]["finetune"]["timestamps"].append(time.time())
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CoWA={coa:.3f}"
    )

# ----------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
