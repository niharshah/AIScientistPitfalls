import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# -------------------- EXPERIMENT DATA STRUCTURE --------------------- #
experiment_data = {
    "FreezeTokenEmbeddings": {  # ablation name
        "SPR_BENCH": {"results": {}}  # dataset name  # one entry per nhead
    }
}

# ---------------------- MISC SETUP ---------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- DATA UTILS ----------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def build_vocab(dataset: Dataset, seq_field: str = "sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    tokens = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    if max_len is not None:
        tokens = tokens[:max_len]
    return tokens


# ----------------------- SYNTHETIC DATA ----------------------------- #
def build_synthetic(num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen_split(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            label = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(label)
        return Dataset.from_dict(data)

    return DatasetDict(
        train=gen_split(num_train), dev=gen_split(num_dev), test=gen_split(num_test)
    )


# ----------------------------- MODEL -------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_flt = (~mask).unsqueeze(-1)
        h_sum = (h * mask_flt).sum(1)
        lengths = mask_flt.sum(1).clamp(min=1)
        pooled = h_sum / lengths
        return self.classifier(pooled)


# ---------------------- DATALOADER UTILS ---------------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len_batch = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (max_len_batch - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, dataloader, criterion):
    model.eval()
    tot_loss, correct, cnt = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            cnt += batch["labels"].size(0)
    return tot_loss / cnt, correct / cnt


# --------------------------- LOAD DATA ------------------------------ #
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

# --------------------- HYPERPARAMETER SWEEP ------------------------- #
nhead_values = [2, 4, 8, 16]
epochs = 5
embed_dim = 128

for nhead in nhead_values:
    if embed_dim % nhead != 0:
        print(f"Skipping nhead={nhead} because embed_dim {embed_dim} not divisible.")
        continue
    print(f"\n=== FreezeTokenEmbeddings | Training with nhead={nhead} ===")
    model = SimpleTransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=2,
        num_classes=num_classes,
        pad_idx=vocab["<pad>"],
    ).to(device)

    # -------------------- FREEZE TOKEN EMBEDDINGS ------------------- #
    for param in model.embed.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    metrics = {"train_acc": [], "val_acc": []}
    losses = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        for batch in train_dl:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
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
        print(
            f"Epoch {epoch}/{epochs} | nhead={nhead} | train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        losses["train_loss"].append(train_loss)
        losses["val_loss"].append(val_loss)

    # ------------------ TEST EVALUATION ----------------------------- #
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"nhead={nhead} | Test accuracy: {test_acc:.4f}")

    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            batch_gpu = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            gts_all.extend(batch["labels"].tolist())

    # --------------- SAVE RESULTS TO EXPERIMENT_DATA ---------------- #
    experiment_data["FreezeTokenEmbeddings"]["SPR_BENCH"]["results"][str(nhead)] = {
        "metrics": metrics,
        "losses": losses,
        "test_acc": test_acc,
        "predictions": preds_all,
        "ground_truth": gts_all,
    }

# ---------------- SAVE ALL EXPERIMENT DATA -------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
