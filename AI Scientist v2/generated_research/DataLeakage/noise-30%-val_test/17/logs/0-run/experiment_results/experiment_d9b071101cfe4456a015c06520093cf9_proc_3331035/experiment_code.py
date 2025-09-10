import os, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from typing import Dict, List
from datasets import DatasetDict, load_dataset

# --------------------------- paths / device -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------- load SPR-BENCH -------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr)


# --------------------------- vocabulary & encode --------------------------
def build_vocab(dsets) -> Dict[str, int]:
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    return {ch: i + 1 for i, ch in enumerate(sorted(chars))}  # 0 = PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())


def encode_sequence(seq: str) -> List[int]:
    return [vocab[ch] for ch in seq]


def pad(seq_ids: List[int]) -> List[int]:
    return (
        seq_ids[:max_len] + [0] * (max_len - len(seq_ids))
        if len(seq_ids) < max_len
        else seq_ids[:max_len]
    )


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.labels = hf_split["label"]
        self.seqs = hf_split["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(pad(encode_sequence(self.seqs[idx])), dtype=torch.long)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"input_ids": ids, "labels": lbl}


train_ds, dev_ds, test_ds = (SPRTorchDataset(spr[x]) for x in ["train", "dev", "test"])
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# --------------------------- model ----------------------------------------
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


criterion = nn.BCEWithLogitsLoss()
weight_decays = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

# --------------------------- experiment data ------------------------------
experiment_data = {"weight_decay_tuning": {}}


def train_one_model(wd: float, epochs: int = 5):
    model = GRUBaseline(vocab_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    tr_losses, tr_mcc, vl_losses, vl_mcc = [], [], [], []
    for ep in range(epochs):
        # training
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item() * batch["labels"].size(0)
        tr_losses.append(running_loss / len(train_ds))
        # compute train MCC quickly
        with torch.no_grad():
            preds, lbls = [], []
            for b in train_loader:
                b = {k: v.to(device) for k, v in b.items()}
                preds.append((model(b["input_ids"]).sigmoid() > 0.5).cpu().numpy())
                lbls.append(b["labels"].cpu().numpy())
        tr_mcc.append(matthews_corrcoef(np.concatenate(lbls), np.concatenate(preds)))
        # validation
        model.eval()
        val_loss = 0.0
        preds, lbls = [], []
        with torch.no_grad():
            for b in dev_loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(b["input_ids"])
                val_loss += criterion(logits, b["labels"]).item() * b["labels"].size(0)
                preds.append((logits.sigmoid() > 0.5).cpu().numpy())
                lbls.append(b["labels"].cpu().numpy())
        vl_losses.append(val_loss / len(dev_ds))
        vl_mcc.append(matthews_corrcoef(np.concatenate(lbls), np.concatenate(preds)))
        print(
            f"wd={wd}  epoch={ep+1}  train_loss={tr_losses[-1]:.4f}  "
            f"val_loss={vl_losses[-1]:.4f}  val_MCC={vl_mcc[-1]:.4f}"
        )
    return model, {
        "metrics": {"train": tr_mcc, "val": vl_mcc},
        "losses": {"train": tr_losses, "val": vl_losses},
    }


best_val_mcc, best_wd, best_model = -1.0, None, None
for wd in weight_decays:
    model, logs = train_one_model(wd)
    experiment_data["weight_decay_tuning"][str(wd)] = logs
    current_val_mcc = logs["metrics"]["val"][-1]
    if current_val_mcc > best_val_mcc:
        best_val_mcc, best_wd, best_model = current_val_mcc, wd, model

print(f"\nBest weight_decay = {best_wd}  with val_MCC = {best_val_mcc:.4f}")

# --------------------------- final test eval ------------------------------
best_model = best_model.to(device).eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = best_model(batch["input_ids"])
        preds_all.append((logits.sigmoid() > 0.5).cpu().numpy())
        labels_all.append(batch["labels"].cpu().numpy())
preds_all = np.concatenate(preds_all)
labels_all = np.concatenate(labels_all)
test_mcc = matthews_corrcoef(labels_all, preds_all)
print(f"Test MCC with best weight_decay: {test_mcc:.4f}")

# store test data in the same dict
experiment_data["weight_decay_tuning"]["best_wd"] = best_wd
experiment_data["weight_decay_tuning"]["predictions"] = preds_all
experiment_data["weight_decay_tuning"]["ground_truth"] = labels_all
experiment_data["weight_decay_tuning"]["test_mcc"] = test_mcc

# --------------------------- save -----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
