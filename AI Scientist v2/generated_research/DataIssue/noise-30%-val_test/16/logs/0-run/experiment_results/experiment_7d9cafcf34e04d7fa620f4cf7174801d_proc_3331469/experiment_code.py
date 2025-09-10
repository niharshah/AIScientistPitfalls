import os, pathlib, random, time, json, math, sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

# --------------------------- experiment data container -----------------------
experiment_data = {
    "num_epochs_tuning": {
        "SPR_BENCH": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
exp_key = experiment_data["num_epochs_tuning"]["SPR_BENCH"]

# --------------------------- working dir -------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------- device ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------- dataset loading ---------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")
        return load_spr_bench(root)

    # ---------- synthetic fallback ----------
    print("Real dataset not found, generating synthetic dataset ...")

    def synth_split(n):
        syms = list("ABCDEFGH")
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(5, 12)
            s = "".join(random.choice(syms) for _ in range(length))
            label = int(s.count("A") % 2 == 0)
            seqs.append(s)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    from datasets import Dataset as HFDataset

    d = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        d[split] = HFDataset.from_dict(synth_split(n))
    return d


spr_bench = maybe_load_real_dataset()
print("Dataset splits:", spr_bench.keys())

# --------------------------- vocabulary --------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved for PAD
itos = {i: ch for ch, i in stoi.items()}
pad_idx = 0
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq: str):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# --------------------------- torch Dataset -----------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = (SPRTorch(spr_bench[s]) for s in ["train", "dev", "test"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)


# --------------------------- model -------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.rnn(e)
        pooled = o.mean(dim=1)
        return self.fc(pooled).squeeze(1)


model = CharBiLSTM(len(vocab)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------------- training with early stopping --------------------
max_epochs = 30
patience = 5
best_val_mcc = -1.0
pat_ctr = 0
best_state = None

for epoch in range(1, max_epochs + 1):
    # ---- training ----
    model.train()
    tr_loss, tr_preds, tr_truths = 0.0, [], []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * batch["x"].size(0)
        tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        tr_truths.extend(batch["y"].cpu().numpy())

    train_loss = tr_loss / len(train_ds)
    train_mcc = matthews_corrcoef(tr_truths, tr_preds)

    # ---- validation ----
    model.eval()
    val_loss_sum, val_preds, val_truths = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            val_loss_sum += loss.item() * batch["x"].size(0)
            val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            val_truths.extend(batch["y"].cpu().numpy())
    val_loss = val_loss_sum / len(val_ds)
    val_mcc = matthews_corrcoef(val_truths, val_preds)

    # ---- logging ----
    exp_key["losses"]["train"].append(train_loss)
    exp_key["losses"]["val"].append(val_loss)
    exp_key["metrics"]["train_MCC"].append(train_mcc)
    exp_key["metrics"]["val_MCC"].append(val_mcc)
    exp_key["epochs"].append(epoch)

    print(
        f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_MCC {val_mcc:.4f}"
    )

    # ---- early stopping ----
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_state = model.state_dict()
        pat_ctr = 0
    else:
        pat_ctr += 1
        if pat_ctr >= patience:
            print(
                f"No improvement for {patience} epochs, stopping early at epoch {epoch}."
            )
            break

# --------------------------- test evaluation ---------------------------------
model.load_state_dict(best_state)
model.eval()
t_preds, t_truths = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["x"])
        t_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        t_truths.extend(batch["y"].cpu().numpy())
test_mcc = matthews_corrcoef(t_truths, t_preds)
print(f"Test MCC (best model): {test_mcc:.4f}")

exp_key["metrics"]["test_MCC"] = test_mcc
exp_key["predictions"] = t_preds
exp_key["ground_truth"] = t_truths

# --------------------------- save & plots ------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

epochs_range = exp_key["epochs"]
plt.figure()
plt.plot(epochs_range, exp_key["losses"]["train"], label="train_loss")
plt.plot(epochs_range, exp_key["losses"]["val"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

plt.figure()
plt.plot(epochs_range, exp_key["metrics"]["val_MCC"], label="val_MCC")
plt.xlabel("epoch")
plt.ylabel("MCC")
plt.title("Validation MCC")
plt.legend()
plt.savefig(os.path.join(working_dir, "mcc_curve.png"))
