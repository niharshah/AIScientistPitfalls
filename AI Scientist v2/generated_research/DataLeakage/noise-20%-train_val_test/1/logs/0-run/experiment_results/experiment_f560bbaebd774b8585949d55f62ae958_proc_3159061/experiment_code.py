import os, pathlib, time, numpy as np, torch, math, gc
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# experiment data container ---------------------------------------------------#
experiment_data = {
    "d_model_tuning": {
        "SPR_BENCH": {
            # will be filled with one sub-dict per d_model setting
        }
    }
}

# -----------------------------------------------------------------------------#
# directories / device --------------------------------------------------------#
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
    req = {"train.csv", "dev.csv", "test.csv"}
    for c in candidates:
        if c and c.exists() and req.issubset({p.name for p in c.iterdir()}):
            print("Found SPR_BENCH at:", c)
            return c
    raise FileNotFoundError("SPR_BENCH dataset not found. Set env variable.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # always use split="train" to load the whole file
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ("train", "dev", "test"):
        dset[split] = _load(f"{split}.csv")
    return dset


# -----------------------------------------------------------------------------#
# PyTorch dataset -------------------------------------------------------------#
class SPRCharDataset(Dataset):
    def __init__(self, hf_dataset, vocab):
        self.data, self.vocab = hf_dataset, vocab
        self.pad_id = vocab["<pad>"]

    def __len__(self):
        return len(self.data)

    def _enc(self, seq: str):
        seq = seq.replace(" ", "")
        return torch.tensor([self.vocab[ch] for ch in seq], dtype=torch.long)

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

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.embed(input_ids) + self.pos_embed[:seq_len, :].unsqueeze(0)
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = x.transpose(0, 1)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(pooled)


# -----------------------------------------------------------------------------#
# training / eval loops -------------------------------------------------------#
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(out, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            correct += (out.argmax(1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return total_loss / total, correct / total


# -----------------------------------------------------------------------------#
# prepare data loaders (done once) --------------------------------------------#
data_dir = _find_spr_bench()
spr = load_spr_bench(data_dir)
print("Loaded splits:", list(spr.keys()))
vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))
max_len = max(len(ex["sequence"].replace(" ", "")) for ex in spr["train"])
print("Max seq len:", max_len)

train_ds = SPRCharDataset(spr["train"], vocab)
dev_ds = SPRCharDataset(spr["dev"], vocab)
test_ds = SPRCharDataset(spr["test"], vocab)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
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

num_labels = len({int(ex["label"]) for ex in spr["train"]})

# -----------------------------------------------------------------------------#
# d_model hyper-parameter sweep -----------------------------------------------#
d_model_values = [32, 64, 128, 256]
epochs = 10
for dm in d_model_values:
    print(f"\n=== Training with d_model={dm} ===")
    tag = f"d_model_{dm}"
    experiment_data["d_model_tuning"]["SPR_BENCH"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = SimpleTransformerClassifier(len(vocab), num_labels, max_len, d_model=dm).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = run_epoch(model, dev_loader, criterion, optimizer=None)
        print(
            f"Ep{epoch:02d} - train_loss:{t_loss:.4f} val_loss:{v_loss:.4f} val_acc:{v_acc*100:.2f}%"
        )
        experiment_data["d_model_tuning"]["SPR_BENCH"][tag]["losses"]["train"].append(
            t_loss
        )
        experiment_data["d_model_tuning"]["SPR_BENCH"][tag]["losses"]["val"].append(
            v_loss
        )
        experiment_data["d_model_tuning"]["SPR_BENCH"][tag]["metrics"]["train"].append(
            t_acc
        )
        experiment_data["d_model_tuning"]["SPR_BENCH"][tag]["metrics"]["val"].append(
            v_acc
        )

    # final test
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
    print(f"Test acc for d_model={dm}: {test_acc*100:.2f}%")

    # store predictions / ground truth
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(1).cpu().numpy()
            experiment_data["d_model_tuning"]["SPR_BENCH"][tag]["predictions"].extend(
                preds
            )
            experiment_data["d_model_tuning"]["SPR_BENCH"][tag]["ground_truth"].extend(
                batch["labels"].cpu().numpy()
            )

    # free memory before next run
    del model, optimizer, criterion, logits, preds
    torch.cuda.empty_cache()
    gc.collect()

# -----------------------------------------------------------------------------#
# save results ----------------------------------------------------------------#
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    f"All experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
)
