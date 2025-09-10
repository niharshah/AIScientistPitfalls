import os, pathlib, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# directories / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------#
# locate SPR_BENCH
def _find_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path(os.getenv("SPR_DATASET_PATH", "")),
        pathlib.Path("./SPR_BENCH").resolve(),
        pathlib.Path("../SPR_BENCH").resolve(),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH").resolve(),
    ]
    files = {"train.csv", "dev.csv", "test.csv"}
    for c in candidates:
        if c and c.exists() and files.issubset({p.name for p in c.iterdir()}):
            print(f"Found SPR_BENCH at: {c}")
            return c
    raise FileNotFoundError(
        "Could not locate SPR_BENCH.  Set SPR_DATA or SPR_DATASET_PATH env var."
    )


# -----------------------------------------------------------------------------#
# dataset loading helper
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ("train", "dev", "test")})


# -----------------------------------------------------------------------------#
# PyTorch dataset
class SPRCharDataset(Dataset):
    def __init__(self, hf_dataset, vocab):
        self.data, self.vocab, self.pad_id = hf_dataset, vocab, vocab["<pad>"]

    def __len__(self):
        return len(self.data)

    def _encode(self, seq: str):
        return torch.tensor(
            [self.vocab[ch] for ch in seq.replace(" ", "")], dtype=torch.long
        )

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": self._encode(row["sequence"]),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def build_vocab(train_split):
    chars = {ch for ex in train_split for ch in ex["sequence"].replace(" ", "")}
    vocab = {"<pad>": 0}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab


def collate_fn(batch, pad_id=0):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attn_mask = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attn_mask, "labels": labels}


# -----------------------------------------------------------------------------#
# model
class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        max_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        pos = self.pos_embed[: input_ids.size(1)].unsqueeze(0)
        x = (self.embed(input_ids) + pos).transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool()).transpose(0, 1)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(pooled)


# -----------------------------------------------------------------------------#
# train / eval loops
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    epoch_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return epoch_loss / total, correct / total


# -----------------------------------------------------------------------------#
# experiment data container
experiment_data = {
    "weight_decay": {
        "SPR_BENCH": {
            "settings": [],  # weight_decay values
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "test_acc": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# -----------------------------------------------------------------------------#
# pipeline
data_dir = _find_spr_bench()
spr = load_spr_bench(data_dir)
print("Loaded SPR_BENCH splits:", list(spr.keys()))
vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")
max_len = max(len(ex["sequence"].replace(" ", "")) for ex in spr["train"])
print(f"Max sequence length in training set: {max_len}")

# datasets / loaders
train_ds, dev_ds, test_ds = (
    SPRCharDataset(spr[s], vocab) for s in ("train", "dev", "test")
)
train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_id=vocab["<pad>"]),
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, pad_id=vocab["<pad>"]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, pad_id=vocab["<pad>"]),
)

criterion = nn.CrossEntropyLoss()
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
num_labels = len(set(int(ex["label"]) for ex in spr["train"]))
epochs = 10

# collect ground truth once
for batch in test_loader:
    experiment_data["weight_decay"]["SPR_BENCH"]["ground_truth"].extend(
        batch["labels"].numpy()
    )
# -----------------------------------------------------------------------------#
for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    experiment_data["weight_decay"]["SPR_BENCH"]["settings"].append(wd)
    model = SimpleTransformerClassifier(len(vocab), num_labels, max_len=max_len).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = run_epoch(model, dev_loader, criterion)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        print(
            f"Epoch {epoch}: train_loss={t_loss:.4f}, val_loss={v_loss:.4f}, "
            f"val_acc={v_acc*100:.2f}%"
        )

    experiment_data["weight_decay"]["SPR_BENCH"]["losses"]["train"].append(train_losses)
    experiment_data["weight_decay"]["SPR_BENCH"]["losses"]["val"].append(val_losses)
    experiment_data["weight_decay"]["SPR_BENCH"]["metrics"]["train"].append(train_accs)
    experiment_data["weight_decay"]["SPR_BENCH"]["metrics"]["val"].append(val_accs)

    # final test evaluation
    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    experiment_data["weight_decay"]["SPR_BENCH"]["test_acc"].append(test_acc)
    print(f"Test accuracy (weight_decay={wd}): {test_acc*100:.2f}%")

    # store predictions
    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds_list.extend(logits.argmax(1).cpu().numpy())
    experiment_data["weight_decay"]["SPR_BENCH"]["predictions"].append(preds_list)

# -----------------------------------------------------------------------------#
# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir, "experiment_data.npy")}')
