import os, pathlib, random, time, json, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

# --- working dir -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment data container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# --- device ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- helper to load SPR_BENCH -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def maybe_load_real_dataset() -> DatasetDict:
    env_path = os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    root = pathlib.Path(env_path)
    if root.exists():
        print(f"Loading real SPR_BENCH from {root}")
        return load_spr_bench(root)
    # ----------------- fallback synthetic data --------------------------
    print("Real dataset not found, generating synthetic toy dataset...")

    def synth_split(n):
        seqs, labels = [], []
        syms = list("ABCDEFGH")
        for _ in range(n):
            length = random.randint(5, 12)
            seq = "".join(random.choice(syms) for _ in range(length))
            label = int(seq.count("A") % 2 == 0)  # simple parity rule on 'A'
            seqs.append(seq)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    dset = DatasetDict()
    dset["train"] = load_dataset(
        "json", data_files={"train": []}, split="train"
    )  # dummy init
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        tmp = synth_split(n)
        dset[split] = load_dataset("json", data_files={"train": []}, split="train")
        dset[split] = dset[split].add_item(tmp)
    # HuggingFace Dataset concat hack – simpler: convert manually:
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        d = synth_split(n)
        dset[split] = load_dataset("csv", data_files={"train": []}, split="train")
    # Easier: build by hand into Dataset.from_dict
    from datasets import Dataset as HFDataset

    dset = DatasetDict()
    for split, n in [("train", 2000), ("dev", 500), ("test", 500)]:
        dset[split] = HFDataset.from_dict(synth_split(n))
    return dset


spr_bench = maybe_load_real_dataset()
print("Loaded splits:", spr_bench.keys())

# --- vocabulary --------------------------------------------------------------
all_text = "".join(spr_bench["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved for PAD
itos = {i: ch for ch, i in stoi.items()}
pad_idx = 0
max_len = min(40, max(len(s) for s in spr_bench["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    return ids


# --- PyTorch dataset ---------------------------------------------------------
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


# --- model -------------------------------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        emb = self.emb(x)  # (B, L, E)
        out, _ = self.lstm(emb)  # (B, L, 2H)
        pooled = out.mean(dim=1)  # simple mean pool
        logits = self.fc(pooled).squeeze(1)  # (B,)
        return logits


model = CharBiLSTM(len(vocab)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- training loop -----------------------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, preds, truths = 0.0, [], []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["x"].size(0)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
        truths.extend(batch["y"].detach().cpu().numpy())
    train_loss = running_loss / len(train_ds)
    train_mcc = matthews_corrcoef(truths, preds)

    # ---- validation ----
    model.eval()
    val_loss_term, v_preds, v_truths = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            val_loss_term += loss.item() * batch["x"].size(0)
            v_preds.extend(torch.sigmoid(logits).cpu().numpy() > 0.5)
            v_truths.extend(batch["y"].cpu().numpy())
    val_loss = val_loss_term / len(val_ds)
    val_mcc = matthews_corrcoef(v_truths, v_preds)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_MCC"].append(train_mcc)
    experiment_data["SPR_BENCH"]["metrics"]["val_MCC"].append(val_mcc)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MCC={val_mcc:.4f}"
    )

# --- final evaluation on test ------------------------------------------------
test_loader = DataLoader(test_ds, batch_size=256)
model.eval()
t_preds, t_truths = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["x"])
        t_preds.extend(torch.sigmoid(logits).cpu().numpy() > 0.5)
        t_truths.extend(batch["y"].cpu().numpy())
test_mcc = matthews_corrcoef(t_truths, t_preds)
print(f"Test MCC: {test_mcc:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["test_MCC"] = test_mcc
experiment_data["SPR_BENCH"]["predictions"] = t_preds
experiment_data["SPR_BENCH"]["ground_truth"] = t_truths

# --- save metrics ------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# --- visualisation -----------------------------------------------------------
epochs_range = experiment_data["SPR_BENCH"]["epochs"]
train_loss = experiment_data["SPR_BENCH"]["losses"]["train"]
val_loss = experiment_data["SPR_BENCH"]["losses"]["val"]
plt.figure()
plt.plot(epochs_range, train_loss, label="train_loss")
plt.plot(epochs_range, val_loss, label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

val_mccs = experiment_data["SPR_BENCH"]["metrics"]["val_MCC"]
plt.figure()
plt.plot(epochs_range, val_mccs, label="val_MCC")
plt.xlabel("epoch")
plt.ylabel("MCC")
plt.legend()
plt.title("Validation MCC")
plt.savefig(os.path.join(working_dir, "mcc_curve.png"))
