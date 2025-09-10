import os, pathlib, random, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------- WORKING DIR & DEVICE ---------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- EXPERIMENT DATA STRUCT ---------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- DATA HELPERS ---------- #
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


def build_synthetic(n_train=5000, n_dev=1000, n_test=2000, seq_len=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def make_split(n):
        ids, seqs, labs = [], [], []
        for i in range(n):
            s = [random.choice(symbols) for _ in range(seq_len)]
            lab = int(s.count("A") % 2 == 0)  # simple parity rule
            ids.append(str(i))
            seqs.append(" ".join(s))
            labs.append(lab)
        return Dataset.from_dict({"id": ids, "sequence": seqs, "label": labs})

    return DatasetDict(
        train=make_split(n_train), dev=make_split(n_dev), test=make_split(n_test)
    )


def build_vocab(dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    nxt = 2
    for seq in dataset[seq_field]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = nxt
                nxt += 1
    return vocab


def encode_sequence(seq, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


def collate(batch, vocab, max_len=128):
    toks = [encode_sequence(b["sequence"], vocab) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_b = max(len(t) for t in toks)
    max_b = min(max_b, max_len)
    padded = []
    counts = []
    for seq in toks:
        seq = seq[:max_b]
        padded.append(seq + [vocab["<pad>"]] * (max_b - len(seq)))
        cv = np.bincount(seq, minlength=len(vocab))
        counts.append(cv)
    input_ids = torch.tensor(padded, dtype=torch.long)
    attn_mask = input_ids == vocab["<pad>"]
    counts = torch.tensor(np.stack(counts), dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "counts": counts,
        "labels": labels,
    }


# ---------- MODEL ---------- #
class HybridTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.count_proj = nn.Linear(vocab_size, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, ids, mask, counts):
        pos = torch.arange(0, ids.size(1), device=ids.device).unsqueeze(0)
        h = self.token_emb(ids) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1e-5)
        c_feat = self.count_proj(counts)
        feat = torch.cat([pooled, c_feat], dim=-1)
        return self.classifier(feat)


# ---------- DATASET LOADING ---------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic dataset:", e)
    dsets = build_synthetic()

vocab = build_vocab(dsets["train"])
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab size={len(vocab)}, classes={num_classes}")

batch_size = 64
train_dl = DataLoader(
    dsets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate(b, vocab),
)
dev_dl = DataLoader(
    dsets["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)
test_dl = DataLoader(
    dsets["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)

# ---------- TRAINING ---------- #
model = HybridTransformerClassifier(
    vocab_size=len(vocab),
    embed_dim=128,
    nhead=2,
    num_layers=2,
    num_classes=num_classes,
    pad_idx=vocab["<pad>"],
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 8


def run_epoch(dataloader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    preds = []
    gts = []
    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if train:
            optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
        loss = criterion(logits, batch["labels"])
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(dataloader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


for epoch in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train=True)
    val_loss, val_f1, _, _ = run_epoch(dev_dl, train=False)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macroF1 = {val_f1:.4f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

# ---------- TEST EVALUATION ---------- #
test_loss, test_f1, preds, gts = run_epoch(test_dl, train=False)
print(f"Test macro-F1: {test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# ---------- SAVE RESULTS ---------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
