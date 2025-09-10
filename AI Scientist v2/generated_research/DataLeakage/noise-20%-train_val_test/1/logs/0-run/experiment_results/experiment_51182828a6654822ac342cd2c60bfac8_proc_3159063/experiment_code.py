import os, pathlib, time, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------#
# directories / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------#
# locate SPR_BENCH ------------------------------------------------------------#
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
        "Could not locate SPR_BENCH.  Set SPR_DATA or SPR_DATASET_PATH."
    )


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
# dataset helpers -------------------------------------------------------------#
class SPRCharDataset(Dataset):
    def __init__(self, hf_dataset, vocab):
        self.data, self.vocab, self.pad_id = hf_dataset, vocab, vocab["<pad>"]

    def __len__(self):
        return len(self.data)

    def _enc(self, seq):
        return torch.tensor(
            [self.vocab[ch] for ch in seq.replace(" ", "")], dtype=torch.long
        )

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": self._enc(row["sequence"]),
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
    attn = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


# -----------------------------------------------------------------------------#
# model -----------------------------------------------------------------------#
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

    def forward(self, ids, attn_mask):
        pos = self.pos_embed[: ids.size(1)].unsqueeze(0)
        x = self.embed(ids) + pos  # B,S,E
        x = x.transpose(0, 1)  # S,B,E
        x = self.encoder(x, src_key_padding_mask=~attn_mask.bool())
        x = x.transpose(0, 1)  # B,S,E
        pooled = (x * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1, keepdim=True)
        return self.classifier(pooled)


# -----------------------------------------------------------------------------#
# train / eval loops -----------------------------------------------------------#
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return total_loss / total, correct / total


# -----------------------------------------------------------------------------#
# load data -------------------------------------------------------------------#
data_dir = _find_spr_bench()
spr = load_spr_bench(data_dir)
print("Loaded splits:", list(spr.keys()))
vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))
max_len = max(len(ex["sequence"].replace(" ", "")) for ex in spr["train"])
train_ds = SPRCharDataset(spr["train"], vocab)
dev_ds = SPRCharDataset(spr["dev"], vocab)
test_ds = SPRCharDataset(spr["test"], vocab)
dev_loader = DataLoader(
    dev_ds,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)

# -----------------------------------------------------------------------------#
# hyper-parameter tuning: batch_size ------------------------------------------#
batch_sizes = [32, 128, 256]
epochs = 10
experiment_data = {"batch_size": {}}

for bs in batch_sizes:
    print(f"\n===== Training with batch size {bs} =====")
    exp_key = f"bs_{bs}"
    experiment_data["batch_size"][exp_key] = {
        "metrics": {"train_acc": [], "val_acc": []},
        "losses": {"train_loss": [], "val_loss": []},
        "predictions": [],
        "ground_truth": [],
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
    )
    model = SimpleTransformerClassifier(
        len(vocab), len(set(int(x["label"]) for x in spr["train"])), max_len=max_len
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, dev_loader, criterion)
        print(
            f"Epoch {ep:2d}/{epochs} | bs {bs:3d} | "
            f"train_loss {tr_loss:.4f} train_acc {tr_acc:.3f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.3f}"
        )
        ed = experiment_data["batch_size"][exp_key]
        ed["losses"]["train_loss"].append(tr_loss)
        ed["losses"]["val_loss"].append(val_loss)
        ed["metrics"]["train_acc"].append(tr_acc)
        ed["metrics"]["val_acc"].append(val_acc)

    # final test evaluation + predictions
    tst_loss, tst_acc = run_epoch(model, test_loader, criterion)
    print(f"Batch size {bs} | Test accuracy: {tst_acc*100:.2f}%")
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(1).cpu().numpy()
            experiment_data["batch_size"][exp_key]["predictions"].extend(preds)
            experiment_data["batch_size"][exp_key]["ground_truth"].extend(
                batch["labels"].cpu().numpy()
            )

# -----------------------------------------------------------------------------#
# save results ----------------------------------------------------------------#
save_path = os.path.join(working_dir, "experiment_data.npy")
np.save(save_path, experiment_data)
print(f"\nAll experiment data saved to {save_path}")
