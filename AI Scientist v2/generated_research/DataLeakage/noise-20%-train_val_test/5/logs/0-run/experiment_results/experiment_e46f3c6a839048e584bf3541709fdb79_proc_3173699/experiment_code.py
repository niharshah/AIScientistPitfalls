# NoTransformerEncoder ablation – single-file script
import os, pathlib, random, math, json, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# -------------------- EXPERIMENT DATA STRUCTURE -------------------- #
experiment_data = {
    "NoTransformerEncoder": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "test_acc": None,
        }
    }
}

# --------------------------- MISC SETUP ---------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------- DATA UTILS -------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
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


def build_vocab(dataset: Dataset, seq_field: str = "sequence"):
    vocab, idx = {"<pad>": 0, "<unk>": 1}, 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq: str, vocab: dict, max_len=None):
    tok_ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    return tok_ids[:max_len] if max_len else tok_ids


# ----------------------- SYNTHETIC DATA ---------------------------- #
def build_synthetic(num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def _gen(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            label = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(label)
        return Dataset.from_dict(data)

    return DatasetDict(train=_gen(num_train), dev=_gen(num_dev), test=_gen(num_test))


# ----------------------------- MODEL ------------------------------- #
class BagOfEmbedsClassifier(nn.Module):
    """NoTransformerEncoder ablation model: EMBED -> MEAN POOL -> FC."""

    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(512, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, pad_mask):
        # x: [B, T], pad_mask True where PAD
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)  # [B,T,E]
        mask_flt = (~pad_mask).unsqueeze(-1).float()  # [B,T,1]
        h_sum = (h * mask_flt).sum(1)  # [B,E]
        lengths = mask_flt.sum(1).clamp(min=1)  # [B,1]
        pooled = h_sum / lengths  # [B,E]
        return self.classifier(pooled)  # [B,C]


# ------------------------- DATALOADER UTILS ------------------------ #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    lbls = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len_b = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (max_len_b - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    pad_mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": pad_mask, "labels": lbls}


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    tot_loss, correct, count = 0.0, 0, 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        tot_loss += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        count += batch["labels"].size(0)
    return tot_loss / count, correct / count


# --------------------------- LOAD DATA ----------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Could not load real dataset, using synthetic:", e)
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

# ----------------------------- TRAIN ------------------------------- #
embed_dim = 128
epochs = 5
model = BagOfEmbedsClassifier(
    vocab_size=len(vocab),
    embed_dim=embed_dim,
    num_classes=num_classes,
    pad_idx=vocab["<pad>"],
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(1, epochs + 1):
    model.train()
    ep_loss, correct, total = 0.0, 0, 0
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    train_loss = ep_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(model, dev_dl, criterion)
    print(f"Epoch {ep}/{epochs} | train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    experiment_data["NoTransformerEncoder"]["SPR_BENCH"]["metrics"]["train"].append(
        train_acc
    )
    experiment_data["NoTransformerEncoder"]["SPR_BENCH"]["metrics"]["val"].append(
        val_acc
    )
    experiment_data["NoTransformerEncoder"]["SPR_BENCH"]["losses"]["train"].append(
        train_loss
    )
    experiment_data["NoTransformerEncoder"]["SPR_BENCH"]["losses"]["val"].append(
        val_loss
    )

# -------------------------- FINAL TEST ----------------------------- #
test_loss, test_acc = evaluate(model, test_dl, criterion)
experiment_data["NoTransformerEncoder"]["SPR_BENCH"]["test_acc"] = test_acc
print(f"Test accuracy (NoTransformerEncoder): {test_acc:.4f}")

# store predictions / ground truth
model.eval()
with torch.no_grad():
    for batch in test_dl:
        batch_gpu = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
        experiment_data["NoTransformerEncoder"]["SPR_BENCH"]["predictions"].extend(
            logits.argmax(-1).cpu().tolist()
        )
        experiment_data["NoTransformerEncoder"]["SPR_BENCH"]["ground_truth"].extend(
            batch["labels"].tolist()
        )

# --------------------------- SAVE DATA ----------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
