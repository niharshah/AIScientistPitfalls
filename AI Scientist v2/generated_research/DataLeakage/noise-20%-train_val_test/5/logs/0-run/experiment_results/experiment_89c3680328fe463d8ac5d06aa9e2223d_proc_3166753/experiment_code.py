import os, pathlib, random, numpy as np, torch, math
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, Dataset, DatasetDict

# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------- EXP-DATA ------------------------------ #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# --------------------- DATASET HELPERS ----------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def build_synthetic(ntr=1000, ndv=200, nte=300, seqlen=12, vocab_sz=10):
    syms = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(syms) for _ in range(seqlen)]
            label = 0 if seq.count("A") % 2 else 1  # parity rule
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(label)
        return Dataset.from_dict(data)

    return DatasetDict(train=gen(ntr), dev=gen(ndv), test=gen(nte))


# -------------------------- VOCAB ---------------------------------- #
def build_vocab(ds, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    for seq in ds[seq_field]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode(seq, vocab, max_len=None):
    toks = [vocab.get(t, vocab["<unk>"]) for t in seq.split()]
    if max_len:
        toks = toks[:max_len]
    return toks


# ----------------------- COLLATE FN -------------------------------- #
def collate(batch, vocab, max_len=128):
    cls_idx = vocab["<cls>"]
    pad_idx = vocab["<pad>"]
    enc = [encode(b["sequence"], vocab, max_len) for b in batch]
    # prepend CLS
    enc = [[cls_idx] + e for e in enc]
    maxL = max(len(e) for e in enc)
    padded = [e + [pad_idx] * (maxL - len(e)) for e in enc]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_idx
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    # bag-of-symbol counts (exclude pad/unk/cls)
    counts = torch.zeros(len(batch), len(vocab), dtype=torch.float32)
    for i, seq in enumerate(enc):
        for t in seq:
            if t > 2:
                counts[i, t] += 1
    return {"input_ids": x, "attention_mask": mask, "counts": counts, "labels": labels}


# --------------------- HYBRID MODEL -------------------------------- #
class HybridTransformer(nn.Module):
    def __init__(
        self, vocab_sz, embed_dim, nhead, n_layers, mlp_hidden, num_classes, pad_idx
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # explicit feature branch
        self.mlp = nn.Sequential(
            nn.Linear(vocab_sz, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x, mask, counts):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        cls_vec = h[:, 0, :]  # [CLS]
        feat = self.mlp(counts)
        logits = self.classifier(torch.cat([cls_vec, feat], dim=-1))
        return logits


# ----------------------- LOAD DATA --------------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH")
except Exception as e:
    print("Falling back to synthetic dataset", e)
    dsets = build_synthetic()

vocab = build_vocab(dsets["train"])
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab size: {len(vocab)}, classes: {num_classes}")

bs = 128
max_len = 128
train_dl = DataLoader(
    dsets["train"],
    batch_size=bs,
    shuffle=True,
    collate_fn=lambda b: collate(b, vocab, max_len),
)
dev_dl = DataLoader(
    dsets["dev"],
    batch_size=bs,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab, max_len),
)
test_dl = DataLoader(
    dsets["test"],
    batch_size=bs,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab, max_len),
)

# ---------------------- TRAINING SETUP ----------------------------- #
embed_dim = 128
nhead = 4
n_layers = 2
mlp_hidden = 64
epochs = 8
model = HybridTransformer(
    len(vocab), embed_dim, nhead, n_layers, mlp_hidden, num_classes, vocab["<pad>"]
).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=3e-4)


def run_eval(dl):
    model.eval()
    ys, preds, losses = [], [], []
    with torch.no_grad():
        for batch in dl:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
            loss = criterion(logits, batch["labels"])
            losses.append(loss.item() * batch["labels"].size(0))
            pred = logits.argmax(-1)
            ys.extend(batch["labels"].cpu().tolist())
            preds.extend(pred.cpu().tolist())
    avg_loss = sum(losses) / len(ys)
    f1 = f1_score(ys, preds, average="macro")
    return avg_loss, f1, preds, ys


# --------------------------- TRAIN LOOP ---------------------------- #
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss, ys_train, preds_train = 0, [], []
    for batch in train_dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optim.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optim.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
        preds_train.extend(logits.argmax(-1).cpu().tolist())
        ys_train.extend(batch["labels"].cpu().tolist())
    train_loss = epoch_loss / len(ys_train)
    train_f1 = f1_score(ys_train, preds_train, average="macro")
    val_loss, val_f1, _, _ = run_eval(dev_dl)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macroF1={val_f1:.4f}")

# ------------------------- TEST EVAL ------------------------------- #
test_loss, test_f1, preds, gts = run_eval(test_dl)
experiment_data["SPR_BENCH"]["metrics"]["test_f1"] = test_f1
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
print(f"Test macro-F1 = {test_f1:.4f}")

# ------------------------- SAVE DATA ------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
