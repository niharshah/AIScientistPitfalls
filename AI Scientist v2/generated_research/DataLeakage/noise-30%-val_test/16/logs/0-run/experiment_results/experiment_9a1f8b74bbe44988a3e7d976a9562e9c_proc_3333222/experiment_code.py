# NUM-epochs hyper-parameter tuning – single file script
import os, pathlib, random, json, math, time
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef


# reproducibility -------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# experiment data container ---------------------------------------------------
experiment_data = {
    "NUM_EPOCHS": {  # <- hyper-parameter tuning type 1
        "SPR_BENCH": {}  # <- dataset name 1
    }
}
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------- dataset loading helpers -----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print("Loading real SPR_BENCH from", root)
        return load_spr_bench(root)

    # ---------- synthetic fallback ------------------------------------------
    print("Real dataset not found – generating synthetic data.")
    from datasets import Dataset as HFDataset

    def synth_split(n):
        syms = list("ABCDEFGH")
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(5, 12)
            seq = "".join(random.choice(syms) for _ in range(length))
            labels.append(int(seq.count("A") % 2 == 0))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    dset = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        dset[split] = HFDataset.from_dict(synth_split(n))
    return dset


spr_bench = maybe_load_real_dataset()

# ------------------------------ vocab ----------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {c: i + 1 for i, c in enumerate(vocab)}  # 0 reserved for PAD
pad_idx, max_len = 0, min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    return ids + [pad_idx] * (max_len - len(ids))


# --------------------------- torch dataset -----------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_dset):
        self.seqs, self.labels = hf_dset["sequence"], hf_dset["label"]

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


# ------------------------------ model ----------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out.mean(1)).squeeze(1)


# ------------------------ training function ----------------------------------
def run_training(max_epochs, patience=3):
    model = CharBiLSTM(len(vocab)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    rec = {
        "losses": {"train": [], "val": []},
        "metrics": {"train_MCC": [], "val_MCC": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }

    best_val, epochs_no_imp = -1e9, 0
    best_model_state = None

    for epoch in range(1, max_epochs + 1):
        # ---- train ----------------------------------------------------------
        model.train()
        tr_loss, preds, truths = 0.0, [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["x"])
            loss = crit(logits, batch["y"])
            loss.backward()
            optim.step()
            tr_loss += loss.item() * batch["x"].size(0)
            preds.extend((torch.sigmoid(logits).detach() > 0.5).cpu().numpy())
            truths.extend(batch["y"].cpu().numpy())
        tr_loss /= len(train_ds)
        tr_mcc = matthews_corrcoef(truths, preds)

        # ---- validation ----------------------------------------------------
        model.eval()
        v_loss, v_preds, v_truths = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["x"])
                loss = crit(logits, batch["y"])
                v_loss += loss.item() * batch["x"].size(0)
                v_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                v_truths.extend(batch["y"].cpu().numpy())
        v_loss /= len(val_ds)
        v_mcc = matthews_corrcoef(v_truths, v_preds)

        # ---- record --------------------------------------------------------
        rec["losses"]["train"].append(tr_loss)
        rec["losses"]["val"].append(v_loss)
        rec["metrics"]["train_MCC"].append(tr_mcc)
        rec["metrics"]["val_MCC"].append(v_mcc)
        rec["epochs"].append(epoch)

        print(
            f"[{max_epochs}-ep] Epoch {epoch}: "
            f"train_loss={tr_loss:.4f} val_loss={v_loss:.4f} val_MCC={v_mcc:.4f}"
        )

        # ---- early stopping ------------------------------------------------
        if v_mcc > best_val:
            best_val, epochs_no_imp, best_model_state = v_mcc, 0, model.state_dict()
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                break

    # ------------------- test with best model -------------------------------
    model.load_state_dict(best_model_state)
    model.eval()
    t_preds, t_truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            t_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            t_truths.extend(batch["y"].cpu().numpy())
    test_mcc = matthews_corrcoef(t_truths, t_preds)
    rec["metrics"]["test_MCC"] = test_mcc
    rec["predictions"], rec["ground_truth"] = t_preds, t_truths
    print(f"[{max_epochs}-ep] Test MCC: {test_mcc:.4f}")
    return rec


# --------------------- hyper-parameter grid search ---------------------------
EPOCH_CHOICES = [5, 10, 20, 30]
best_overall, best_cfg = -1e9, None
for ep in EPOCH_CHOICES:
    record = run_training(ep, patience=3)
    experiment_data["NUM_EPOCHS"]["SPR_BENCH"][f"epochs_{ep}"] = record
    val_best = max(record["metrics"]["val_MCC"])
    if val_best > best_overall:
        best_overall, best_cfg = val_best, f"epochs_{ep}"

print("Best configuration:", best_cfg, "best val_MCC:", best_overall)

# ------------------------- plotting best run ---------------------------------
best_run = experiment_data["NUM_EPOCHS"]["SPR_BENCH"][best_cfg]
epochs_range = best_run["epochs"]
plt.figure()
plt.plot(epochs_range, best_run["losses"]["train"], label="train_loss")
plt.plot(epochs_range, best_run["losses"]["val"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(working_dir, f"loss_curve_{best_cfg}.png"))

plt.figure()
plt.plot(epochs_range, best_run["metrics"]["val_MCC"], label="val_MCC")
plt.xlabel("epoch")
plt.ylabel("MCC")
plt.legend()
plt.title("Validation MCC")
plt.savefig(os.path.join(working_dir, f"mcc_curve_{best_cfg}.png"))

# -------------------------- save experiment data -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
