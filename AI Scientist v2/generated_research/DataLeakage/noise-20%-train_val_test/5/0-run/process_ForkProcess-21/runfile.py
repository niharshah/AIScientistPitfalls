# Remove-FFN ablation – single-file runnable script
import os, pathlib, random, math, json, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ----------------- EXPERIMENT DATA STRUCTURE ----------------- #
experiment_data = {
    "RemoveFeedForwardNetwork": {
        "SPR_BENCH": {"results": {}}  # one entry per nhead value
    }
}

# ------------------ MISC / DEVICE --------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------- DATASET HELPERS ------------------------ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def build_vocab(dataset: Dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset[seq_field]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    return toks[:max_len] if max_len else toks


def build_synthetic(n_train=500, n_dev=100, n_test=200, seqlen=10, vocab_sz=12):
    syms = [chr(ord("A") + i) for i in range(vocab_sz)]

    def _gen(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(syms) for _ in range(seqlen)]
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(1 if seq.count("A") % 2 == 0 else 0)
        return Dataset.from_dict(d)

    return DatasetDict(train=_gen(n_train), dev=_gen(n_dev), test=_gen(n_test))


# ----------------- MODEL COMPONENTS ------------------------ #
class TransformerEncoderLayerNoFFN(nn.Module):
    """Encoder layer without the position-wise FFN (identity mapping)."""

    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout1(attn_output))
        return src


class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        nhead,
        num_layers,
        num_classes,
        pad_idx,
        remove_ffn=False,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(512, embed_dim)
        if remove_ffn:
            layer_cls = lambda: TransformerEncoderLayerNoFFN(
                embed_dim, nhead, dropout=0.1, batch_first=True
            )
        else:
            layer_cls = lambda: nn.TransformerEncoderLayer(
                embed_dim,
                nhead,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            )
        self.encoder = nn.TransformerEncoder(layer_cls(), num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_inv = (~mask).unsqueeze(-1)  # 1 for real tokens
        summed = (h * mask_inv).sum(1)
        lengths = mask_inv.sum(1).clamp(min=1)
        pooled = summed / lengths
        return self.classifier(pooled)


# ------------------- COLLATE & EVAL ------------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    maxlen = max(len(s) for s in seqs)
    pad_id = vocab["<pad>"]
    padded = [s + [pad_id] * (maxlen - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_id
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            pred = logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return tot_loss / total, correct / total


# ------------------- LOAD DATA ------------------------------------ #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic:", e)
    dsets = build_synthetic()

vocab = build_vocab(dsets["train"])
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab size: {len(vocab)} | num_classes: {num_classes}")

batch_size = 64
train_dl = DataLoader(
    dsets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab),
)
dev_dl = DataLoader(
    dsets["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)
test_dl = DataLoader(
    dsets["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)

# ------------------- TRAINING SWEEP ------------------------------- #
embed_dim, epochs = 128, 5
nhead_values = [2, 4, 8, 16]

for nhead in nhead_values:
    if embed_dim % nhead != 0:
        print(f"Skip nhead={nhead} – embed_dim not divisible.")
        continue
    print(f"\n=== nhead {nhead} (FFN removed) ===")
    model = SimpleTransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=2,
        num_classes=num_classes,
        pad_idx=vocab["<pad>"],
        remove_ffn=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics, losses = {"train_acc": [], "val_acc": []}, {
        "train_loss": [],
        "val_loss": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optim.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            pred = logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
        train_loss, train_acc = tot_loss / total, correct / total
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        print(f"Epoch {ep}/{epochs} | train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        losses["train_loss"].append(train_loss)
        losses["val_loss"].append(val_loss)

    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"nhead={nhead} | Test accuracy: {test_acc:.4f}")

    # Predictions / ground truth for test set
    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            gts_all.extend(batch["labels"].tolist())

    experiment_data["RemoveFeedForwardNetwork"]["SPR_BENCH"]["results"][str(nhead)] = {
        "metrics": metrics,
        "losses": losses,
        "test_acc": test_acc,
        "predictions": np.array(preds_all),
        "ground_truth": np.array(gts_all),
    }

# ------------------- SAVE ALL DATA -------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
