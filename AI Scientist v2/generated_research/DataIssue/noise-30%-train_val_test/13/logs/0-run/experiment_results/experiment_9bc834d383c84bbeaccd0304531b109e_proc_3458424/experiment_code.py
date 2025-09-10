# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, math, time, json, random

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment data container --------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------- dataset load (provided helper adjusted) --------------------
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
    raise FileNotFoundError("SPR_BENCH folder not found in expected locations.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocabulary --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = set()
for s in spr["train"]["sequence"]:
    vocab.update(list(s))
vocab = [PAD, UNK] + sorted(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# -------------------- label mapping --------------------
labels = sorted(list(set(spr["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Num classes: {num_classes}")

# -------------------- encoding helpers --------------------
MAX_LEN = 64


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in list(seq)[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def encode_label(lab):
    return label2id[lab]


# -------------------- PyTorch dataset --------------------
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
train_dl = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# -------------------- model --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        mask = input_ids == 0  # pad mask
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        return self.fc(x)


model = TransformerClassifier(vocab_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------- training --------------------
def run_epoch(dataloader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in dataloader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        all_preds.append(logits.argmax(-1).detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, macro_f1, all_preds, all_labels


EPOCHS = 8
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_f1, _, _ = run_epoch(train_dl, train=True)
    val_loss, val_f1, _, _ = run_epoch(val_dl, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}  val_macroF1={val_f1:.4f}  (train_loss={train_loss:.4f})  [{time.time()-t0:.1f}s]"
    )

# -------------------- final test evaluation --------------------
test_loss, test_f1, test_preds, test_labels = run_epoch(test_dl, train=False)
experiment_data["SPR_BENCH"]["predictions"] = test_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels.tolist()
print(f"Test macro-F1: {test_f1:.4f}")

# -------------------- save experiment data --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
