import os, pathlib, math, time, json, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- experiment container --------------------
experiment_data = {
    "num_epochs": {  # hyper-parameter tuning type
        "SPR_BENCH": {  # dataset name
            # each of the following will become a list-of-lists keyed by run id
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],  # store the epoch schedule e.g. [1..12]
            "epoch_config": [],  # store num_epochs used for the run
        }
    }
}

# -------------------- device & reproducibility --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------- dataset load --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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


data_root_candidates = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
for p in data_root_candidates:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocabulary --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = set()
for s in spr["train"]["sequence"]:
    vocab.update(s)
vocab = [PAD, UNK] + sorted(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

# -------------------- labels --------------------
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)

# -------------------- helpers --------------------
MAX_LEN = 64


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def encode_label(lab):
    return label2id[lab]


# -------------------- torch dataset --------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labs = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(encode_label(self.labs[idx]), dtype=torch.long),
        }


batch_size = 128
train_ds = SPRTorchDataset(spr["train"])
val_ds = SPRTorchDataset(spr["dev"])
test_ds = SPRTorchDataset(spr["test"])

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# -------------------- model --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)


# -------------------- train / eval functions --------------------
def run_epoch(model, dataloader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    macro_f1 = f1_score(labels, preds, average="macro")
    return total_loss / len(dataloader.dataset), macro_f1, preds, labels


# -------------------- hyper-parameter sweep --------------------
epoch_options = [8, 12, 16, 24]
for run_idx, max_epochs in enumerate(epoch_options):
    print(f"\n=== Run {run_idx+1}/{len(epoch_options)} | num_epochs={max_epochs} ===")
    model = TransformerClassifier(vocab_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_train_losses, run_val_losses = [], []
    run_train_f1s, run_val_f1s = [], []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_dl, criterion)
        run_train_losses.append(tr_loss)
        run_val_losses.append(val_loss)
        run_train_f1s.append(tr_f1)
        run_val_f1s.append(val_f1)
        print(
            f"Epoch {epoch}/{max_epochs}  val_loss={val_loss:.4f}  val_F1={val_f1:.4f}  "
            f"(train_loss={tr_loss:.4f}) [{time.time()-t0:.1f}s]"
        )

    # test evaluation after training
    test_loss, test_f1, test_preds, test_labels = run_epoch(model, test_dl, criterion)

    # -------------------- log results --------------------
    ed = experiment_data["num_epochs"]["SPR_BENCH"]
    ed["losses"]["train"].append(run_train_losses)
    ed["losses"]["val"].append(run_val_losses)
    ed["metrics"]["train_f1"].append(run_train_f1s)
    ed["metrics"]["val_f1"].append(run_val_f1s)
    ed["predictions"].append(test_preds.tolist())
    ed["ground_truth"].append(test_labels.tolist())
    ed["epochs"].append(list(range(1, max_epochs + 1)))
    ed["epoch_config"].append(max_epochs)
    print(f"Run completed. Test macro-F1: {test_f1:.4f}")

# -------------------- save experiment --------------------
np.save("experiment_data.npy", experiment_data, allow_pickle=True)
print("\nSaved experiment_data.npy")
