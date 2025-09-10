# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, json, math, time
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef

# ------------------------ experiment container ------------------------------
experiment_data = {
    "LEARNING_RATE": {"SPR_BENCH": {}}  # each lr will get its own dict here
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------- reproducibility ------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ----------------------------- device ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------- dataset loading / synthesis --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _l("train.csv")
    d["dev"] = _l("dev.csv")
    d["test"] = _l("test.csv")
    return d


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print("Loading real SPR_BENCH from", root)
        return load_spr_bench(root)
    print("Real dataset not found → generating synthetic data")
    from datasets import Dataset as HFDataset

    syms = list("ABCDEFGH")

    def synth_split(n):
        seqs, labs = [], []
        for _ in range(n):
            ln = random.randint(5, 12)
            seq = "".join(random.choice(syms) for _ in range(ln))
            lab = int(seq.count("A") % 2 == 0)  # parity on 'A'
            seqs.append(seq)
            labs.append(lab)
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    ddict = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        ddict[split] = HFDataset.from_dict(synth_split(n))
    return ddict


spr_bench = maybe_load_real_dataset()
print("Splits:", spr_bench.keys())

# -------------------------- vocabulary --------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 == PAD
itos = {i: ch for ch, i in stoi.items()}
pad_idx = 0
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    return ids


class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labs = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return {
            "x": torch.tensor(encode(self.seqs[i]), dtype=torch.long),
            "y": torch.tensor(self.labs[i], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = (SPRTorch(spr_bench[s]) for s in ["train", "dev", "test"])


# ----------------------------- model ----------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out.mean(1)).squeeze(1)


# ------------------------ training util -------------------------------------
def run_training(lr, epochs=5, batch=128):
    tr_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=256)
    model = CharBiLSTM(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    rec = {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
    }

    for ep in range(1, epochs + 1):
        # ---- training ----
        model.train()
        tloss = 0
        tp, tt = [], []
        for b in tr_loader:
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            logits = model(b["x"])
            loss = crit(logits, b["y"])
            loss.backward()
            opt.step()
            tloss += loss.item() * b["x"].size(0)
            tp += list((torch.sigmoid(logits).detach().cpu().numpy()) > 0.5)
            tt += list(b["y"].cpu().numpy())
        tr_loss = tloss / len(train_ds)
        tr_mcc = matthews_corrcoef(tt, tp)

        # ---- validation ----
        model.eval()
        vloss = 0
        vp, vt = [], []
        with torch.no_grad():
            for b in va_loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(b["x"])
                loss = crit(logits, b["y"])
                vloss += loss.item() * b["x"].size(0)
                vp += list((torch.sigmoid(logits).cpu().numpy()) > 0.5)
                vt += list(b["y"].cpu().numpy())
        va_loss = vloss / len(val_ds)
        va_mcc = matthews_corrcoef(vt, vp)

        rec["losses"]["train"].append(tr_loss)
        rec["losses"]["val"].append(va_loss)
        rec["metrics"]["train_MCC"].append(tr_mcc)
        rec["metrics"]["val_MCC"].append(va_mcc)
        rec["epochs"].append(ep)
        print(
            f"lr {lr:.0e} | epoch {ep}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, val_MCC={va_mcc:.4f}"
        )
    rec["model_state"] = model.state_dict()  # save weights for best later
    return rec


# -------------------------- hyperparam sweep --------------------------------
sweep_lrs = [3e-4, 5e-4, 1e-3, 2e-3]
best_lr, best_val = None, -1
for lr in sweep_lrs:
    record = run_training(lr)
    experiment_data["LEARNING_RATE"]["SPR_BENCH"][f"{lr:.0e}"] = record
    top_val = max(record["metrics"]["val_MCC"])
    if top_val > best_val:
        best_val, best_lr = top_val, lr
print(f"Best LR: {best_lr:.0e} with peak val_MCC={best_val:.4f}")

# -------------------------- test evaluation ---------------------------------
best_state = experiment_data["LEARNING_RATE"]["SPR_BENCH"][f"{best_lr:.0e}"][
    "model_state"
]
best_model = CharBiLSTM(len(vocab)).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_loader = DataLoader(test_ds, batch_size=256)
tp, tt = [], []
with torch.no_grad():
    for b in test_loader:
        b = {k: v.to(device) for k, v in b.items()}
        logits = best_model(b["x"])
        tp += list((torch.sigmoid(logits).cpu().numpy()) > 0.5)
        tt += list(b["y"].cpu().numpy())
test_mcc = matthews_corrcoef(tt, tp)
print("Test MCC with best lr:", test_mcc)
experiment_data["LEARNING_RATE"]["SPR_BENCH"]["best_lr"] = best_lr
experiment_data["LEARNING_RATE"]["SPR_BENCH"]["test_MCC"] = test_mcc
experiment_data["LEARNING_RATE"]["SPR_BENCH"]["predictions"] = tp
experiment_data["LEARNING_RATE"]["SPR_BENCH"]["ground_truth"] = tt

# ----------------------------- save -----------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ---------------------------- plotting --------------------------------------
best_rec = experiment_data["LEARNING_RATE"]["SPR_BENCH"][f"{best_lr:.0e}"]
ep = best_rec["epochs"]
plt.figure()
plt.plot(ep, best_rec["losses"]["train"], label="train")
plt.plot(ep, best_rec["losses"]["val"], label="val")
plt.title(f"Loss curve (best lr={best_lr:.0e})")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.figure()
plt.plot(ep, best_rec["metrics"]["val_MCC"])
plt.title("Validation MCC")
plt.xlabel("epoch")
plt.ylabel("MCC")
plt.savefig(os.path.join(working_dir, "mcc_curve.png"))
