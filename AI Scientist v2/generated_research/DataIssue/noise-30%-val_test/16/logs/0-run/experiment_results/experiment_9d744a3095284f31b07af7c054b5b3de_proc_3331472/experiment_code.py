# dropout_hparam_tuning.py
import os, pathlib, random, time, json, math
import numpy as np
import torch, sys
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef

# ---------- experiment store -------------------------------------------------
experiment_data = {"dropout_tuning": {}}  # each sub-key will be the dropout rate tried

# ---------- device -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- dataset helpers --------------------------------------------------
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


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")
        return load_spr_bench(root)

    # --------- fallback: very small synthetic toy task -----------------------
    print("Real dataset not found, generating synthetic data…")

    def synth_split(n):
        seqs, labels = [], []
        alphabet = list("ABCDEFGH")
        for _ in range(n):
            length = random.randint(5, 12)
            seq = "".join(random.choice(alphabet) for _ in range(length))
            label = int(seq.count("A") % 2 == 0)
            seqs.append(seq)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    from datasets import Dataset as HFDataset, DatasetDict

    dset = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        dset[split] = HFDataset.from_dict(synth_split(n))
    return dset


spr_bench = maybe_load_real_dataset()

# ---------- vocab / encoding -------------------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD
itos = {i: ch for ch, i in stoi.items()}
pad_idx = 0
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids[:max_len]


# ---------- torch datasets ---------------------------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_dataset):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


train_ds = SPRTorch(spr_bench["train"])
val_ds = SPRTorch(spr_bench["dev"])
test_ds = SPRTorch(spr_bench["test"])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)


# ---------- model definition -------------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        emb = self.emb(x)  # (B, L, E)
        out, _ = self.lstm(emb)  # (B, L, 2H)
        pooled = out.mean(dim=1)  # (B, 2H)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(1)  # (B,)
        return logits


# ---------- hyper-parameter sweep --------------------------------------------
dropout_rates = [0.0, 0.2, 0.5]
num_epochs = 5
criterion = nn.BCEWithLogitsLoss()

for d_rate in dropout_rates:
    print(f"\n=== Training with dropout={d_rate} ===")
    # init model / optimiser fresh each run
    model = CharBiLSTM(len(vocab), dropout=d_rate).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # prepare experiment_data slot
    exp_key = str(d_rate)
    experiment_data["dropout_tuning"][exp_key] = {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, num_epochs + 1):
        # ----- train -----
        model.train()
        t_loss, t_preds, t_gts = 0.0, [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optim.step()

            t_loss += loss.item() * batch["x"].size(0)
            t_preds.extend((torch.sigmoid(logits) > 0.5).detach().cpu().numpy())
            t_gts.extend(batch["y"].cpu().numpy())
        train_loss = t_loss / len(train_ds)
        train_mcc = matthews_corrcoef(t_gts, t_preds)

        # ----- validation -----
        model.eval()
        v_loss, v_preds, v_gts = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["x"])
                loss = criterion(logits, batch["y"])
                v_loss += loss.item() * batch["x"].size(0)
                v_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                v_gts.extend(batch["y"].cpu().numpy())
        val_loss = v_loss / len(val_ds)
        val_mcc = matthews_corrcoef(v_gts, v_preds)

        # store
        ed = experiment_data["dropout_tuning"][exp_key]
        ed["losses"]["train"].append(train_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train_MCC"].append(train_mcc)
        ed["metrics"]["val_MCC"].append(val_mcc)
        ed["epochs"].append(epoch)

        print(
            f"  epoch {epoch}: "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} "
            f"val_MCC={val_mcc:.4f}"
        )

    # ----- final test evaluation ----------
    model.eval()
    test_preds, test_gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            test_gts.extend(batch["y"].cpu().numpy())
    test_mcc = matthews_corrcoef(test_gts, test_preds)
    experiment_data["dropout_tuning"][exp_key]["metrics"]["test_MCC"] = test_mcc
    experiment_data["dropout_tuning"][exp_key]["predictions"] = test_preds
    experiment_data["dropout_tuning"][exp_key]["ground_truth"] = test_gts
    print(f"  ==> Test MCC (dropout={d_rate}): {test_mcc:.4f}")

# ---------- save all experiment data -----------------------------------------
out_dir = os.path.join(os.getcwd(), "working")
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "experiment_data.npy"), experiment_data)
print("\nSaved results to", os.path.join(out_dir, "experiment_data.npy"))
