# RemovePositionalEmbeddings – single-file ablation study
# ======================================================
import os, pathlib, random, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ---------------- EXPERIMENT DATA STRUCT ---------------- #
experiment_data = {
    "RemovePositionalEmbeddings": {
        "SPR_BENCH": {  # will also be used for synthetic when real data missing
            "results": {}
        }
    }
}

# ---------------------- SETUP --------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- DATA UTILITIES -------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


def build_vocab(dataset: Dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq: str, vocab, max_len=None):
    toks = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    return toks[:max_len] if max_len else toks


# ------------------ SYNTHETIC BACK-UP ------------------- #
def build_synthetic(n_train=500, n_dev=100, n_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(1 if seq.count("A") % 2 == 0 else 0)
        return Dataset.from_dict(data)

    return DatasetDict(train=gen(n_train), dev=gen(n_dev), test=gen(n_test))


# -------------------- MODEL W/O POS ---------------------- #
class TransformerNoPos(nn.Module):
    """Transformer encoder classifier without positional embeddings."""

    def __init__(self, vocab_sz, embed_dim, nhead, nlayers, n_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Linear(embed_dim, n_classes)

    def forward(self, x, pad_mask):
        h = self.embed(x)  # << NO positional encoding here
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        not_pad = (~pad_mask).unsqueeze(-1).float()
        h_sum = (h * not_pad).sum(1)
        lengths = not_pad.sum(1).clamp(min=1)
        pooled = h_sum / lengths
        return self.cls(pooled)


# -------------------- DATALOADER ------------------------ #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len_b = max(len(s) for s in seqs)
    pad_id = vocab["<pad>"]
    padded = [s + [pad_id] * (max_len_b - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_id
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, correct, n_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            n_samples += batch["labels"].size(0)
    return tot_loss / n_samples, correct / n_samples


# -------------------- LOAD DATA ------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Failed to load real dataset, falling back to synthetic.", e)
    datasets_dict = build_synthetic()

vocab = build_vocab(datasets_dict["train"])
num_classes = len(set(datasets_dict["train"]["label"]))
print("Vocab size:", len(vocab), "| num_classes:", num_classes)

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

# ------------------ HYPERPARAM SWEEP -------------------- #
embed_dim = 128
epochs = 5
nhead_values = [2, 4, 8, 16]

for nhead in nhead_values:
    if embed_dim % nhead != 0:
        print(f"Skip nhead={nhead}, embed_dim not divisible.")
        continue
    print(f"\n=== Ablation: training w/o positional encodings | nhead={nhead} ===")
    model = TransformerNoPos(
        vocab_sz=len(vocab),
        embed_dim=embed_dim,
        nhead=nhead,
        nlayers=2,
        n_classes=num_classes,
        pad_idx=vocab["<pad>"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    metrics, losses = {"train_acc": [], "val_acc": []}, {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        print(
            f"Epoch {epoch}/{epochs} | train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        losses["train_loss"].append(train_loss)
        losses["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

    # --------------- Test evaluation -------------------- #
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"nhead={nhead} | Test accuracy: {test_acc:.4f}")

    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            batch_gpu = {
                k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
            }
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            gts_all.extend(batch["labels"].tolist())

    # -------------- log results ------------------------- #
    experiment_data["RemovePositionalEmbeddings"]["SPR_BENCH"]["results"][
        str(nhead)
    ] = {
        "metrics": metrics,
        "losses": losses,
        "test_acc": test_acc,
        "predictions": preds_all,
        "ground_truth": gts_all,
    }

# -------------------- SAVE RESULTS ---------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
