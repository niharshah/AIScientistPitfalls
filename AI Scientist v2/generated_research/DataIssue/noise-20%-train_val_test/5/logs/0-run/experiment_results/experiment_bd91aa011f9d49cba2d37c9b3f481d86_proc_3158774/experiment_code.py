import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# --------------------- REPRODUCIBILITY --------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------- EXPERIMENT DICT --------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"batch_size_tuning": {"SPR_BENCH": {}}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------- DATA UTILITIES ---------------------- #
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


def build_vocab(dataset: Dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    return toks[:max_len] if max_len else toks


def build_synthetic(num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen_split(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(1 if seq.count("A") % 2 == 0 else 0)
        return Dataset.from_dict(data)

    return DatasetDict(
        train=gen_split(num_train), dev=gen_split(num_dev), test=gen_split(num_test)
    )


# --------------------- MODEL DEFINITION -------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(512, embed_dim)
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
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_flt = (~mask).unsqueeze(-1)
        h_sum = (h * mask_flt).sum(1)
        lengths = mask_flt.sum(1).clamp(min=1)
        pooled = h_sum / lengths
        return self.cls(pooled)


# --------------------- COLLATE FUNCTION -------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len_b = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (max_len_b - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = correct = count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            count += batch["labels"].size(0)
    return tot_loss / count, correct / count


# --------------------- LOAD DATA --------------------------- #
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

# --------------------- HYPERPARAM GRID --------------------- #
batch_sizes = [16, 32, 64, 128, 256]
EPOCHS = 5

for bs in batch_sizes:
    print(f"\n=== Training with batch size {bs} ===")
    # DataLoaders
    train_dl = DataLoader(
        datasets_dict["train"],
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab),
    )
    dev_dl = DataLoader(
        datasets_dict["dev"],
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab),
    )
    test_dl = DataLoader(
        datasets_dict["test"],
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab),
    )
    # Model, criterion, optimizer
    model = SimpleTransformerClassifier(
        len(vocab), 128, 4, 2, num_classes, pad_idx=vocab["<pad>"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage for metrics
    run_record = {
        "metrics": {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = correct = total = 0
        for batch in train_dl:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
        train_loss = epoch_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        print(
            f"Epoch {epoch} | bs {bs} | train_acc {train_acc:.3f} val_acc {val_acc:.3f}"
        )
        run_record["epochs"].append(epoch)
        run_record["metrics"]["train_loss"].append(train_loss)
        run_record["metrics"]["val_loss"].append(val_loss)
        run_record["metrics"]["train_acc"].append(train_acc)
        run_record["metrics"]["val_acc"].append(val_acc)

    # Test evaluation
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"Batch {bs} test_acc: {test_acc:.3f}")
    # predictions
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for batch in test_dl:
            gpu = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(gpu["input_ids"], gpu["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            gts_all.extend(batch["labels"].tolist())
    run_record["predictions"] = preds_all
    run_record["ground_truth"] = gts_all
    run_record["test_acc"] = test_acc
    experiment_data["batch_size_tuning"]["SPR_BENCH"][str(bs)] = run_record

    # clean up
    del model, train_dl, dev_dl, test_dl
    torch.cuda.empty_cache()

# --------------------- SAVE RESULTS ------------------------ #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
