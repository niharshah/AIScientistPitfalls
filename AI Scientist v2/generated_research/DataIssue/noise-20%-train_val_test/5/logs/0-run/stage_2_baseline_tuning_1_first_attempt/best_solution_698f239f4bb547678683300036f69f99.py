import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from datasets import Dataset, DatasetDict, load_dataset
except Exception as _:
    # minimal fallback if `datasets` is unavailable
    class DatasetDict(dict):
        pass

    def load_dataset(*a, **k):
        raise RuntimeError("datasets lib missing")

    from types import SimpleNamespace

    Dataset = SimpleNamespace
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"learning_rate_tuning": {"SPR_BENCH": {}}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- DATA ---------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_synthetic(num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            label = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(label)
        from datasets import Dataset

        return Dataset.from_dict(data)

    from datasets import DatasetDict

    return DatasetDict(train=gen(num_train), dev=gen(num_dev), test=gen(num_test))


try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Could not load real dataset, using synthetic:", e)
    datasets_dict = build_synthetic()


def build_vocab(dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab(datasets_dict["train"])
pad_idx = vocab["<pad>"]
num_classes = len(set(datasets_dict["train"]["label"]))
print(f"Vocab size: {len(vocab)}, classes: {num_classes}")


def encode_sequence(seq, vocab, max_len=None):
    ids = [vocab.get(t, vocab["<unk>"]) for t in seq.strip().split()]
    return ids[:max_len] if max_len else ids


def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    L = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (L - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


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


# --------------------------- MODEL ---------------------------------- #
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
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        pos_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=mask)
        non_pad = (~mask).unsqueeze(-1)
        h_sum = (h * non_pad).sum(1)
        lengths = non_pad.sum(1).clamp(min=1)
        pooled = h_sum / lengths
        return self.fc(pooled)


def evaluate(model, dl, criterion):
    model.eval()
    tot_loss = correct = count = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            count += batch["labels"].size(0)
    return tot_loss / count, correct / count


# ----------------------- TRAIN PER LR -------------------------------- #
def train_one_lr(lr, epochs=5):
    model = SimpleTransformerClassifier(len(vocab), 128, 4, 2, num_classes, pad_idx).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = correct = count = 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            count += batch["labels"].size(0)
        train_loss, train_acc = tot_loss / count, correct / count
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        metrics["epochs"].append(ep)
        print(
            f"LR {lr:.1e} | Epoch {ep}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"LR {lr:.1e} | Test acc {test_acc:.3f}")
    # predictions
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for batch in test_dl:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            gts_all.extend(batch["labels"].tolist())
    return metrics, preds_all, gts_all, test_acc


lr_grid = [5e-4, 1e-3, 5e-3]
for lr in lr_grid:
    metrics, preds, gt, test_acc = train_one_lr(lr)
    lr_key = f"{lr:.1e}"
    experiment_data["learning_rate_tuning"]["SPR_BENCH"][lr_key] = {
        "metrics": metrics,
        "predictions": preds,
        "ground_truth": gt,
        "test_acc": test_acc,
    }

# -------------------- SAVE RESULT ----------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
