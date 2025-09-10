import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
from typing import List

# ---------- saving / folders ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR_BENCH or synth ----------
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


def get_dataset() -> DatasetDict:
    possible = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for p in possible:
        if (p / "train.csv").exists():
            print(f"Loading real SPR_BENCH from {p}")
            return load_spr_bench(p)

    # --- synthetic fallback ---
    def synth(n):
        rows, shapes = [], "ABCD"
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            rows.append(
                {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}
            )
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    print("SPR_BENCH not found, creating synthetic toy dataset")
    return DatasetDict(
        train=to_ds(synth(2000)), dev=to_ds(synth(500)), test=to_ds(synth(500))
    )


spr = get_dataset()

# ---------- vocab ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved for PAD
vocab_size, max_len = len(stoi) + 1, min(100, max(map(len, spr["train"]["sequence"])))


def encode(seq: str) -> List[int]:
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq, self.y = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.float),
        }


batch_size = 128
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size)


# ---------- model ----------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int = 64, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        _, h = self.rnn(self.emb(x))
        h = torch.cat((h[0], h[1]), 1)
        return self.fc(h).squeeze(1)


model = CharBiGRU(vocab_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- experiment tracking ----------
experiment_data = {
    "num_epochs": {  # hyperparameter tuning type
        "SPR_BENCH": {
            "metrics": {"train_macro_f1": [], "val_macro_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
track = experiment_data["num_epochs"]["SPR_BENCH"]

# ---------- training w/ early stopping ----------
max_epochs, patience = 30, 5
best_val, bad_epochs = float("inf"), 0
for epoch in range(1, max_epochs + 1):
    # ---- train ----
    model.train()
    tr_loss, tr_pred, tr_lab = [], [], []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        tr_loss.append(loss.item())
        tr_pred.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
        tr_lab.extend(batch["label"].long().cpu().numpy())
    train_f1 = f1_score(tr_lab, tr_pred, average="macro")

    # ---- validation ----
    model.eval()
    va_loss, va_pred, va_lab = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            va_loss.append(criterion(logits, batch["label"]).item())
            va_pred.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
            va_lab.extend(batch["label"].long().cpu().numpy())
    val_loss = np.mean(va_loss)
    val_f1 = f1_score(va_lab, va_pred, average="macro")

    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_macro_f1={val_f1:.4f}")
    # ---- logging ----
    track["metrics"]["train_macro_f1"].append(train_f1)
    track["metrics"]["val_macro_f1"].append(val_f1)
    track["losses"]["train"].append(np.mean(tr_loss))
    track["losses"]["val"].append(val_loss)
    track["epochs"].append(epoch)

    # ---- early stopping ----
    if val_loss < best_val - 1e-4:
        best_val, bad_epochs = val_loss, 0
    else:
        bad_epochs += 1
    if bad_epochs >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# ---------- test evaluation ----------
model.eval()
test_pred, test_lab = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        test_pred.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
        test_lab.extend(batch["label"].long().cpu().numpy())
test_macro_f1 = f1_score(test_lab, test_pred, average="macro")
print(f"Test Macro-F1: {test_macro_f1:.4f}")

track["predictions"], track["ground_truth"] = test_pred, test_lab

# ---------- save artifacts ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure(figsize=(6, 4))
plt.plot(track["epochs"], track["losses"]["train"], label="train_loss")
plt.plot(track["epochs"], track["losses"]["val"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("SPR_BENCH_loss_curve")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
plt.close()
