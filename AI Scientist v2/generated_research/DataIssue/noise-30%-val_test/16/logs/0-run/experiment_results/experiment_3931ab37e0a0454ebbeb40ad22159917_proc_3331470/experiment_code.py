# hyperparam-tuning_learning_rate.py
import os, pathlib, random, time, json, math, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------- #
# I/O helpers
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "learning_rate": {"SPR_BENCH": {}}  # hyperparam tuning type  # dataset name
}

# --------------------------------------------------------------------------- #
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------------------------------- #
# data loading (identical to baseline)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ("train", "dev", "test"):
        d[split] = _load(f"{split}.csv")
    return d


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print("Loading real SPR_BENCH from", root)
        return load_spr_bench(root)

    print("Real dataset not found – generating toy dataset.")

    def synth_split(n):
        syms = list("ABCDEFGH")
        seqs, labels = [], []
        for _ in range(n):
            s = "".join(random.choice(syms) for _ in range(random.randint(5, 12)))
            labels.append(int(s.count("A") % 2 == 0))
            seqs.append(s)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    from datasets import Dataset as HFDataset

    d = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        d[split] = HFDataset.from_dict(synth_split(n))
    return d


spr_bench = maybe_load_real_dataset()
print("Dataset splits loaded:", spr_bench.keys())

# --------------------------------------------------------------------------- #
# vocabulary util
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
pad_idx, max_len = 0, min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# --------------------------------------------------------------------------- #
# torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_dataset):
        self.s = hf_dataset["sequence"]
        self.y = hf_dataset["label"]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.s[idx]), dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


train_ds, val_ds, test_ds = (SPRTorch(spr_bench[s]) for s in ("train", "dev", "test"))
train_loader = lambda bs: DataLoader(train_ds, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)


# --------------------------------------------------------------------------- #
# model
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.fc(h.mean(1)).squeeze(1)


# --------------------------------------------------------------------------- #
# hyper-parameter grid
lr_grid = [1e-4, 3e-4, 1e-3, 3e-3]
n_epochs = 5
best_val_mcc, best_lr = -1.0, None

for lr in lr_grid:
    print(f"\n--- Training with learning-rate = {lr} ---")
    run_key = f"{lr:.0e}"
    experiment_data["learning_rate"]["SPR_BENCH"][run_key] = {
        "metrics": {"train_MCC": [], "val_MCC": [], "test_MCC": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    run_store = experiment_data["learning_rate"]["SPR_BENCH"][run_key]

    model = CharBiLSTM(len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        t_loss, t_pred, t_true = 0.0, [], []
        for batch in train_loader(128):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optim.step()

            t_loss += loss.item() * batch["x"].size(0)
            t_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            t_true.extend(batch["y"].cpu().numpy())

        train_loss = t_loss / len(train_ds)
        train_mcc = matthews_corrcoef(t_true, t_pred)

        # validation
        model.eval()
        v_loss, v_pred, v_true = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["x"])
                v_loss += criterion(logits, batch["y"]).item() * batch["x"].size(0)
                v_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                v_true.extend(batch["y"].cpu().numpy())
        val_loss = v_loss / len(val_ds)
        val_mcc = matthews_corrcoef(v_true, v_pred)

        # store
        run_store["losses"]["train"].append(train_loss)
        run_store["losses"]["val"].append(val_loss)
        run_store["metrics"]["train_MCC"].append(train_mcc)
        run_store["metrics"]["val_MCC"].append(val_mcc)
        run_store["epochs"].append(epoch)

        print(
            f"  epoch {epoch}: train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_MCC={val_mcc:.4f}"
        )

    # test evaluation
    model.eval()
    t_pred, t_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            t_pred.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            t_true.extend(batch["y"].cpu().numpy())
    test_mcc = matthews_corrcoef(t_true, t_pred)
    run_store["metrics"]["test_MCC"] = test_mcc
    run_store["predictions"] = t_pred
    run_store["ground_truth"] = t_true
    print(f"  ==> Test MCC @ lr {lr} : {test_mcc:.4f}")

    # track best
    if run_store["metrics"]["val_MCC"][-1] > best_val_mcc:
        best_val_mcc = run_store["metrics"]["val_MCC"][-1]
        best_lr = lr
        best_run_key = run_key
        best_model_state = model.state_dict()

print(f"\nBest learning-rate: {best_lr} (val_MCC={best_val_mcc:.4f})")

# --------------------------------------------------------------------------- #
# optional: plot curves for best run
best = experiment_data["learning_rate"]["SPR_BENCH"][best_run_key]
plt.figure()
plt.plot(best["epochs"], best["losses"]["train"], label="train_loss")
plt.plot(best["epochs"], best["losses"]["val"], label="val_loss")
plt.title(f"Loss curves (lr={best_lr})")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(os.path.join(working_dir, f"loss_lr_{best_run_key}.png"))

plt.figure()
plt.plot(best["epochs"], best["metrics"]["val_MCC"], label="val_MCC")
plt.title(f"Validation MCC (lr={best_lr})")
plt.xlabel("epoch")
plt.ylabel("MCC")
plt.legend()
plt.savefig(os.path.join(working_dir, f"mcc_lr_{best_run_key}.png"))

# --------------------------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
