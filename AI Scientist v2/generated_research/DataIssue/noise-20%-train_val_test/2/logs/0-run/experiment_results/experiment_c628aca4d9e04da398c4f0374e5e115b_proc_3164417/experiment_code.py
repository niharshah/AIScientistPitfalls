import os, pathlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------------
# working directory & device
# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# experiment data container
# -------------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": [], "test_macro_f1": None},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------------------------------------------------------------
# dataset utilities
# -------------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]
        self.max_len = max_len
        self.vocab_size = len(vocab)

    def __len__(self):
        return len(self.seqs)

    def _seq_to_ids(self, seq):
        ids = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in seq[: self.max_len]]
        ids += [self.pad_id] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        ids = self._seq_to_ids(seq)
        # histogram / count vector
        count_vec = np.zeros(self.vocab_size, dtype=np.float32)
        for ch in seq:
            count_vec[self.vocab.get(ch, self.vocab["<unk>"])] += 1.0
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "count_vec": torch.tensor(count_vec, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# -------------------------------------------------------------------------
# hybrid model : transformer + count pathway
# -------------------------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(
        self, vocab_size, num_classes, d_model=128, nhead=4, num_layers=2, max_len=128
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.count_proj = nn.Linear(vocab_size, d_model)
        self.cls = nn.Linear(d_model * 2, num_classes)

    def forward(self, input_ids, count_vec):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.encoder(x).mean(dim=1)  # (batch, d_model)
        c = F.relu(self.count_proj(count_vec))  # (batch, d_model)
        h = torch.cat([x, c], dim=1)  # (batch, 2*d_model)
        return self.cls(h)


# -------------------------------------------------------------------------
# helper : train / eval epoch
# -------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    tot_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["input_ids"], batch["count_vec"])
        loss = criterion(out, batch["labels"])
        loss.backward()
        optimizer.step()

        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    f1 = f1_score(trues, preds, average="macro")
    return tot_loss / len(loader.dataset), f1


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    tot_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(batch["input_ids"], batch["count_vec"])
        loss = criterion(out, batch["labels"])
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    f1 = f1_score(trues, preds, average="macro")
    return tot_loss / len(loader.dataset), f1, preds, trues


# -------------------------------------------------------------------------
# main procedure (executes immediately)
# -------------------------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Please adjust DATA_PATH.")

dset = load_spr_bench(DATA_PATH)

# Build vocabulary
chars = set("".join(dset["train"]["sequence"]))
vocab = {"<pad>": 0, "<unk>": 1}
vocab.update({ch: i + 2 for i, ch in enumerate(sorted(chars))})
vocab_size = len(vocab)

max_len = min(128, max(len(s) for s in dset["train"]["sequence"]))

# DataLoaders
batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(dset["train"], vocab, max_len), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    SPRTorchDataset(dset["dev"], vocab, max_len), batch_size=256, shuffle=False
)
test_loader = DataLoader(
    SPRTorchDataset(dset["test"], vocab, max_len), batch_size=256, shuffle=False
)

num_classes = len(set(dset["train"]["label"]))
model = HybridModel(vocab_size, num_classes, max_len=max_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_f1, patience, wait, best_state = 0.0, 5, 0, None
max_epochs = 15

for epoch in range(1, max_epochs + 1):
    tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
    )

    # record metrics
    ed = experiment_data["SPR_BENCH"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_macro_f1"].append(tr_f1)
    ed["metrics"]["val_macro_f1"].append(val_f1)

    # early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        wait = 0
        best_state = model.state_dict()
    else:
        wait += 1
    if wait >= patience:
        print(
            f"Early stopping triggered at epoch {epoch}. Best val_macro_f1 = {best_val_f1:.4f}"
        )
        break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

# Test evaluation
test_loss, test_f1, preds, gts = eval_epoch(model, test_loader, criterion)
print(f"Test set macro_F1 = {test_f1:.4f}")

ed = experiment_data["SPR_BENCH"]
ed["losses"]["test"] = test_loss
ed["metrics"]["test_macro_f1"] = test_f1
ed["predictions"] = preds
ed["ground_truth"] = gts

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
