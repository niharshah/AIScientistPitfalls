# ------------------------------------------------------------
# Hyper-parameter tuning for DROPOUT_PROB on SPR_BENCH
# ------------------------------------------------------------
import os, pathlib, random, time, json, math, warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef

# ---------- working dir & experiment container -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "DROPOUT_PROB": {  # hyper-parameter tuning category
        "SPR_BENCH": {  # dataset name
            "dropouts": {}  # will be filled with individual runs
        }
    }
}

# ---------- device -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
    for sp in [("train", "train.csv"), ("dev", "dev.csv"), ("test", "test.csv")]:
        d[sp[0]] = _load(sp[1])
    return d


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")
        return load_spr_bench(root)

    # --------- synthetic fallback -----------------
    print("Real dataset not found; generating synthetic toy data ...")

    def synth_split(n):
        seqs, labels = [], []
        vocab_syms = list("ABCDEFGH")
        for _ in range(n):
            l = random.randint(5, 12)
            seq = "".join(random.choice(vocab_syms) for _ in range(l))
            label = int(seq.count("A") % 2 == 0)
            seqs.append(seq)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    from datasets import Dataset as HFDataset

    dset = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        dset[split] = HFDataset.from_dict(synth_split(n))
    return dset


spr_bench = maybe_load_real_dataset()
print("Loaded splits:", spr_bench.keys())

# ---------- vocabulary & encoding --------------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
pad_idx = 0
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# ---------- PyTorch dataset --------------------------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_set):
        self.seq = hf_set["sequence"]
        self.lab = hf_set["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "y": torch.tensor(self.lab[idx], dtype=torch.float32),
        }


train_ds = SPRTorch(spr_bench["train"])
val_ds = SPRTorch(spr_bench["dev"])
test_ds = SPRTorch(spr_bench["test"])


# ---------- model ------------------------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        pooled = out.mean(1)
        pooled = self.dp(pooled)
        return self.fc(pooled).squeeze(1)


# ---------- training routine -------------------------------------------------
def run_experiment(dropout_prob, epochs=5, lr=1e-3, batch_size=128):
    print(f"\n=== Training with dropout={dropout_prob} ===")
    run_record = {
        "epochs": [],
        "losses": {"train": [], "val": []},
        "metrics": {"train_MCC": [], "val_MCC": [], "test_MCC": None},
        "predictions": [],
        "ground_truth": [],
    }

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    tst_loader = DataLoader(test_ds, batch_size=256)

    model = CharBiLSTM(len(vocab), dropout=dropout_prob).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t_loss, t_preds, t_truth = 0.0, [], []
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch["x"])
            loss = crit(logits, batch["y"])
            loss.backward()
            opt.step()
            t_loss += loss.item() * batch["x"].size(0)
            t_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            t_truth.extend(batch["y"].cpu().numpy())
        tr_loss = t_loss / len(train_ds)
        tr_mcc = matthews_corrcoef(t_truth, t_preds)

        # ---- validation ----
        model.eval()
        v_loss, v_preds, v_truth = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["x"])
                loss = crit(logits, batch["y"])
                v_loss += loss.item() * batch["x"].size(0)
                v_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                v_truth.extend(batch["y"].cpu().numpy())
        val_loss = v_loss / len(val_ds)
        val_mcc = matthews_corrcoef(v_truth, v_preds)

        run_record["epochs"].append(ep)
        run_record["losses"]["train"].append(tr_loss)
        run_record["losses"]["val"].append(val_loss)
        run_record["metrics"]["train_MCC"].append(tr_mcc)
        run_record["metrics"]["val_MCC"].append(val_mcc)

        print(
            f"epoch {ep}: tr_loss={tr_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_MCC={val_mcc:.4f}"
        )

    # ---- final test ----
    model.eval()
    tst_preds, tst_truth = [], []
    with torch.no_grad():
        for batch in tst_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            tst_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tst_truth.extend(batch["y"].cpu().numpy())
    test_mcc = matthews_corrcoef(tst_truth, tst_preds)
    run_record["metrics"]["test_MCC"] = test_mcc
    run_record["predictions"] = tst_preds
    run_record["ground_truth"] = tst_truth
    print(f"Test MCC for dropout={dropout_prob}: {test_mcc:.4f}")
    return run_record


# ---------- run experiments ---------------------------------------------------
dropout_values = [0.0, 0.2, 0.4, 0.6]
for d in dropout_values:
    record = run_experiment(d)
    experiment_data["DROPOUT_PROB"]["SPR_BENCH"]["dropouts"][str(d)] = record

# ---------- save experiment data ---------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------- visualisation -----------------------------------------------------
for d in dropout_values:
    rec = experiment_data["DROPOUT_PROB"]["SPR_BENCH"]["dropouts"][str(d)]
    ep = rec["epochs"]
    plt.figure()
    plt.plot(ep, rec["losses"]["train"], label="train_loss")
    plt.plot(ep, rec["losses"]["val"], label="val_loss")
    plt.title(f"Loss Curve (dropout={d})")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(working_dir, f"loss_curve_dropout_{d}.png"))
    plt.close()

    plt.figure()
    plt.plot(ep, rec["metrics"]["val_MCC"], label="val_MCC")
    plt.title(f"Validation MCC (dropout={d})")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("MCC")
    plt.savefig(os.path.join(working_dir, f"mcc_curve_dropout_{d}.png"))
    plt.close()
