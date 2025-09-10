import os, pathlib, time, numpy as np, torch, math, random
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
# locate SPR_BENCH  -----------------------------------------------------------#
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
        "Could not locate SPR_BENCH. Set SPR_DATA or SPR_DATASET_PATH env var."
    )


# -----------------------------------------------------------------------------#
# dataset loading helper ------------------------------------------------------#
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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
        self.data = hf_dataset
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]

    def __len__(self):
        return len(self.data)

    def _encode(self, seq: str):
        seq = seq.replace(" ", "")
        return torch.tensor([self.vocab[ch] for ch in seq], dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": self._encode(row["sequence"]),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def build_vocab(train_split):
    chars = set()
    for ex in train_split:
        chars.update(ex["sequence"].replace(" ", ""))
    vocab = {"<pad>": 0}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab


def collate_fn(batch, pad_id=0):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attention_mask = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attention_mask, "labels": labels}


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
        pos = self.pos_embed[:seq_len, :].unsqueeze(0)  # 1,S,E
        x = self.embed(input_ids) + pos  # B,S,E
        x = x.transpose(0, 1)  # S,B,E
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = x.transpose(0, 1)  # B,S,E
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(pooled)


# -----------------------------------------------------------------------------#
# train / eval loops -----------------------------------------------------------#
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    epoch_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(outputs, batch["labels"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
            preds = outputs.argmax(1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return epoch_loss / total, correct / total


# -----------------------------------------------------------------------------#
# main pipeline ---------------------------------------------------------------#
# fix random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data_dir = _find_spr_bench()
spr = load_spr_bench(data_dir)
print("Loaded SPR_BENCH splits:", list(spr.keys()))

vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")
max_len = max(len(ex["sequence"].replace(" ", "")) for ex in spr["train"])
print(f"Max sequence length in training set: {max_len}")

train_ds = SPRCharDataset(spr["train"], vocab)
dev_ds = SPRCharDataset(spr["dev"], vocab)
test_ds = SPRCharDataset(spr["test"], vocab)

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

num_labels = len(set(int(ex["label"]) for ex in spr["train"]))

# -----------------------------------------------------------------------------#
# hyperparameter tuning: nhead -------------------------------------------------#
nhead_values = [2, 4, 8]
epochs = 10
experiment_data = {
    "nhead_tuning": {
        "SPR_BENCH": {
            "configs": [],
            "metrics": {"train_acc": [], "val_acc": [], "test_acc": []},
            "losses": {"train_loss": [], "val_loss": [], "test_loss": []},
            "predictions": [],  # list of np arrays per config
            "ground_truth": [],  # filled once
        }
    }
}

best_val_acc, best_idx = -1, -1

for idx, nh in enumerate(nhead_values):
    print(f"\n### Starting run {idx+1}/{len(nhead_values)} with nhead={nh} ###")
    model = SimpleTransformerClassifier(
        len(vocab), num_labels, max_len=max_len, nhead=nh
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = run_epoch(model, dev_loader, criterion, optimizer=None)
        print(
            f"[nhead={nh}] Epoch {epoch:02d}: train_loss={t_loss:.4f}, val_loss={v_loss:.4f}, val_acc={v_acc*100:.2f}%"
        )

    # after training evaluate on all splits
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer=None)
    val_loss, val_acc = run_epoch(model, dev_loader, criterion, optimizer=None)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)

    # store metrics
    ed = experiment_data["nhead_tuning"]["SPR_BENCH"]
    ed["configs"].append({"nhead": nh})
    ed["metrics"]["train_acc"].append(tr_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["test_acc"].append(test_acc)
    ed["losses"]["train_loss"].append(tr_loss)
    ed["losses"]["val_loss"].append(val_loss)
    ed["losses"]["test_loss"].append(test_loss)

    # store predictions / ground truth
    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds_all.extend(logits.argmax(1).cpu().numpy())
            gts_all.extend(batch["labels"].cpu().numpy())
    ed["predictions"].append(np.array(preds_all))
    if len(ed["ground_truth"]) == 0:
        ed["ground_truth"] = np.array(gts_all)

    if val_acc > best_val_acc:
        best_val_acc, best_idx = val_acc, idx

print(
    "\nValidation accuracies per nhead:",
    experiment_data["nhead_tuning"]["SPR_BENCH"]["metrics"]["val_acc"],
)
print(f"Best nhead={nhead_values[best_idx]} with val_acc={best_val_acc*100:.2f}%")
print(
    f"Corresponding test accuracy: {experiment_data['nhead_tuning']['SPR_BENCH']['metrics']['test_acc'][best_idx]*100:.2f}%"
)

# -----------------------------------------------------------------------------#
# save results ----------------------------------------------------------------#
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")
