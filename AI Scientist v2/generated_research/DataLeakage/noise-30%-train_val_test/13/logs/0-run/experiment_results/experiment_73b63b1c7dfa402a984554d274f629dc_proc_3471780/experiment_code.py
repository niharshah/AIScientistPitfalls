import os, pathlib, time, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None, "SGA": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------------------------------------------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH not found")

spr = load_spr(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------- Vocabulary & encoding ---------------------------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = [PAD, UNK] + sorted({c for s in spr["train"]["sequence"] for c in s})
stoi = {ch: i for i, ch in enumerate(vocab)}
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
vocab_size, num_classes = len(vocab), len(label2id)
MAX_LEN = 64
print(f"vocab_size={vocab_size}, num_classes={num_classes}")


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def count_vector(seq):
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in seq:
        vec[stoi.get(ch, stoi[UNK])] += 1.0
    return vec


# ---------------- Dataset ------------------------------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode_seq(seq), dtype=torch.long),
            "counts": torch.tensor(count_vector(seq), dtype=torch.float),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
        }


batch_size = 128
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# ---------------- Model --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CountAwareTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 256, 0.1, batch_first=True)
        self.trf = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.count_proj = nn.Linear(vocab_size, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids, counts):
        mask = input_ids.eq(0)
        x = self.pos(self.embed(input_ids))
        x = self.trf(x, src_key_padding_mask=mask)
        x.masked_fill_(mask.unsqueeze(-1), 0)
        seq_repr = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        count_repr = self.count_proj(counts)
        feat = torch.cat([seq_repr, count_repr], dim=-1)
        return self.fc(feat)


model = CountAwareTransformer(vocab_size, 128, 4, 2, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


# ---------------- Train / Eval -------------------------------------------------
def run_epoch(dl, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"], batch["counts"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).cpu())
        gts.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    gts = torch.cat(gts).numpy()
    return (
        total_loss / len(dl.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(val_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vl_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(vl_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={vl_loss:.4f} val_F1={vl_f1:.4f} train_F1={tr_f1:.4f}"
    )

# ---------------- Test ---------------------------------------------------------
ts_loss, ts_f1, ts_preds, ts_gts = run_epoch(test_dl)
experiment_data["SPR_BENCH"]["test_f1"] = ts_f1
experiment_data["SPR_BENCH"][
    "SGA"
] = ts_f1  # placeholder until unseen-rule ids provided
experiment_data["SPR_BENCH"]["predictions"] = ts_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = ts_gts.tolist()
print(f"Test macro-F1 = {ts_f1:.4f}")

# ---------------- Save ---------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
