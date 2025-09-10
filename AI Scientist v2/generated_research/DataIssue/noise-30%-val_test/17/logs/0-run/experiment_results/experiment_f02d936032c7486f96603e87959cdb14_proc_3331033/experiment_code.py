import os, pathlib, numpy as np, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from datasets import DatasetDict, load_dataset

# ------------------------- reproducibility -------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------- data loading ----------------------------
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
print("Loaded:", spr)


# ------------------------- vocab / encoding ------------------------
def build_vocab(dsets):
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    return {ch: idx + 1 for idx, ch in enumerate(sorted(chars))}  # 0=PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())


def encode(s):
    return [vocab[ch] for ch in s]


def pad(seq):
    return (
        seq[:max_len] + [0] * (max_len - len(seq))
        if len(seq) < max_len
        else seq[:max_len]
    )


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.labels = hf_split["label"]
        self.seqs = hf_split["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(pad(encode(self.seqs[idx])), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"input_ids": ids, "labels": label}


train_ds, dev_ds, test_ds = (SPRTorchDataset(spr[k]) for k in ["train", "dev", "test"])
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# -------------------------- model ----------------------------------
class GRUWithDropout(nn.Module):
    def __init__(self, vocab_sz, embed_dim=64, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)  # h: [2,B,H]
        h = torch.cat([h[0], h[1]], dim=1)  # [B,2H]
        h = self.drop(h)
        return self.fc(h).squeeze(1)


# -------------------------- training utils -------------------------
def epoch_pass(model, loader, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.append((logits.sigmoid() > 0.5).detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return total_loss / len(loader.dataset), matthews_corrcoef(labels, preds)


# ----------------------- hyperparameter loop -----------------------
dropout_grid = [0.1, 0.3, 0.5]
epochs = 5
criterion = nn.BCEWithLogitsLoss()

experiment_data = {"dropout_rate": {}}
best_val_mcc, best_state, best_p = -1.0, None, None

for p in dropout_grid:
    key = f"p={p}"
    print(f"\n=== Training with dropout {p} ===")
    model = GRUWithDropout(vocab_size, dropout_rate=p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # containers
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mcc = epoch_pass(model, train_loader, optimizer)
        val_loss, val_mcc = epoch_pass(model, dev_loader)
        exp["losses"]["train"].append(tr_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train"].append(tr_mcc)
        exp["metrics"]["val"].append(val_mcc)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_MCC={val_mcc:.4f}"
        )
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_state = model.state_dict()
            best_p = p
    experiment_data["dropout_rate"][key] = exp

# ----------------------- test on best model ------------------------
print(f"\nBest dropout probability: {best_p} with dev MCC {best_val_mcc:.4f}")
best_model = GRUWithDropout(vocab_size, dropout_rate=best_p).to(device)
best_model.load_state_dict(best_state)
best_model.eval()

preds, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = best_model(batch["input_ids"])
        preds.append((logits.sigmoid() > 0.5).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
preds = np.concatenate(preds)
labels = np.concatenate(labels)
test_mcc = matthews_corrcoef(labels, preds)
print(f"Test MCC (best model): {test_mcc:.4f}")

best_key = f"p={best_p}"
experiment_data["dropout_rate"][best_key]["predictions"] = preds
experiment_data["dropout_rate"][best_key]["ground_truth"] = labels
experiment_data["dropout_rate"][best_key]["test_MCC"] = test_mcc

# ----------------------- save --------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
