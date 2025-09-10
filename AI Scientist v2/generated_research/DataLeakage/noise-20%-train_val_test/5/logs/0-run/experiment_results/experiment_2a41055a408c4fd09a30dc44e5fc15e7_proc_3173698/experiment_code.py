# ------------------------------------------------------------
# Single-layer Transformer ablation study – self-contained file
# ------------------------------------------------------------
import os, pathlib, random, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ---------------- EXPERIMENT DATA DICT ---------------------- #
experiment_data = {
    "SingleTransformerLayer": {"SPR_BENCH": {"results": {}}}  # one entry per nhead
}

# -------------------- BASIC SETUP --------------------------- #
work_dir = os.path.join(os.getcwd(), "working")
os.makedirs(work_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------ DATA UTILS ------------------------ #
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


def build_vocab(dataset: Dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for seq in dataset[seq_field]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(t, vocab["<unk>"]) for t in seq.strip().split()]
    return toks[:max_len] if max_len else toks


# -------------- SYNTHETIC DATA FALLBACK --------------------- #
def build_synthetic(n_train=500, n_dev=100, n_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def _gen(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            lab = 1 if seq.count("A") % 2 == 0 else 0
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(lab)
        return Dataset.from_dict(d)

    return DatasetDict(train=_gen(n_train), dev=_gen(n_dev), test=_gen(n_test))


# ---------------------- MODEL ------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        pos_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_inv = (~mask).unsqueeze(-1)
        pooled = (h * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)
        return self.cls(pooled)


# ------------------- BATCH COLLATOR ------------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    maxlen = max(len(s) for s in seqs)
    pad_idx = vocab["<pad>"]
    padded = [s + [pad_idx] * (maxlen - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_idx
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


# -------------------- LOAD DATA ----------------------------- #
DATA_ROOT = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_ROOT)
    print("Loaded SPR_BENCH dataset.")
except Exception as e:
    print("Could not load SPR_BENCH; using synthetic. Err:", e)
    datasets_dict = build_synthetic()

vocab = build_vocab(datasets_dict["train"])
num_classes = len(set(datasets_dict["train"]["label"]))
print(f"Vocab size: {len(vocab)}, num_classes: {num_classes}")

batch_size = 64
train_dl = DataLoader(
    datasets_dict["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab),
)
dev_dl = DataLoader(
    datasets_dict["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)
test_dl = DataLoader(
    datasets_dict["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)


# --------------------- TRAIN / EVAL ------------------------- #
def evaluate(model, dl, crit):
    model.eval()
    tot_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for bt in dl:
            bt = {k: v.to(device) for k, v in bt.items() if isinstance(v, torch.Tensor)}
            out = model(bt["input_ids"], bt["attention_mask"])
            loss = crit(out, bt["labels"])
            preds = out.argmax(-1)
            tot_loss += loss.item() * bt["labels"].size(0)
            correct += (preds == bt["labels"]).sum().item()
            n += bt["labels"].size(0)
    return tot_loss / n, correct / n


# -------------------- ABLATION RUN -------------------------- #
embed_dim = 128
nhead_values = [2, 4, 8, 16]
epochs = 5

for nhead in nhead_values:
    if embed_dim % nhead != 0:
        print(f"Skipping nhead={nhead}; embed_dim not divisible.")
        continue
    print(f"\n=== SingleLayer Ablation | nhead={nhead} ===")
    model = SimpleTransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=1,  # <-- key ablation change
        num_classes=num_classes,
        pad_idx=vocab["<pad>"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0
        correct = 0
        total = 0
        for bt in train_dl:
            bt = {k: v.to(device) for k, v in bt.items() if isinstance(v, torch.Tensor)}
            optim.zero_grad()
            out = model(bt["input_ids"], bt["attention_mask"])
            loss = criterion(out, bt["labels"])
            loss.backward()
            optim.step()
            ep_loss += loss.item() * bt["labels"].size(0)
            preds = out.argmax(-1)
            correct += (preds == bt["labels"]).sum().item()
            total += bt["labels"].size(0)
        train_loss = ep_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        metrics["train"].append(train_acc)
        metrics["val"].append(val_acc)
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)
        print(f"Epoch {ep}/{epochs} | train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # -------------- TEST & LOGGING -------------------------- #
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"nhead={nhead} | TEST ACCURACY: {test_acc:.4f}")

    # predictions / ground truth
    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for bt in test_dl:
            bt_gpu = {
                k: v.to(device) for k, v in bt.items() if isinstance(v, torch.Tensor)
            }
            out = model(bt_gpu["input_ids"], bt_gpu["attention_mask"])
            preds_all.extend(out.argmax(-1).cpu().tolist())
            gts_all.extend(bt["labels"].tolist())

    experiment_data["SingleTransformerLayer"]["SPR_BENCH"]["results"][str(nhead)] = {
        "metrics": metrics,
        "losses": losses,
        "test_acc": test_acc,
        "predictions": preds_all,
        "ground_truth": gts_all,
    }

# ------------------ SAVE EVERYTHING ------------------------- #
save_path = os.path.join(work_dir, "experiment_data.npy")
np.save(save_path, experiment_data)
print("\nSaved experiment data to", save_path)
