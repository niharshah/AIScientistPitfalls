# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ---------- dirs ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- data ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):  # helper to read csv
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def get_dataset() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)

    # ---------- synthetic fallback ----------
    print("SPR_BENCH not found, creating synthetic toy dataset")

    def synth(n):
        rows, shapes = [], "ABCD"
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        to_ds(synth(2000)),
        to_ds(synth(500)),
        to_ds(synth(500)),
    )
    return d


spr = get_dataset()

# ---------- vocab ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 is PAD
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
max_len = min(100, max(map(len, spr["train"]["sequence"])))


def encode(seq: str):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seq, self.y = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.float),
        }


train_loader = lambda s: DataLoader(
    SPRDataset(spr[s]), batch_size=128, shuffle=(s == "train")
)


# ---------- model ----------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        _, h = self.rnn(self.emb(x))
        h = torch.cat([h[0], h[1]], 1)
        h = self.drop(h)
        return self.fc(h).squeeze(1)


# ---------- experiment store ----------
experiment_data = {"dropout_rate": {}}

# ---------- hyper-parameter sweep ----------
dropout_grid = [0.0, 0.1, 0.3, 0.5]
epochs = 5
for rate in dropout_grid:
    print(f"\n=== Training with dropout_rate={rate} ===")
    model = CharBiGRU(vocab_size, dropout=rate).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rec = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
    # ---- epoch loop ----
    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_loss, tr_preds, tr_labels = [], [], []
        for batch in train_loader("train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["label"])
            loss.backward()
            opt.step()
            tr_loss.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_labels.extend(batch["label"].long().cpu().numpy())
        train_f1 = f1_score(tr_labels, tr_preds, average="macro")

        # validate
        model.eval()
        val_loss, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch in train_loader("dev"):
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                val_loss.append(crit(logits, batch["label"]).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_labels.extend(batch["label"].long().cpu().numpy())
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(
            f"Epoch {ep} | train_loss={np.mean(tr_loss):.4f} val_loss={np.mean(val_loss):.4f} "
            f"train_F1={train_f1:.3f} val_F1={val_f1:.3f}"
        )

        rec["metrics"]["train_macro_f1"].append(train_f1)
        rec["metrics"]["val_macro_f1"].append(val_f1)
        rec["losses"]["train"].append(np.mean(tr_loss))
        rec["losses"]["val"].append(np.mean(val_loss))
        rec["epochs"].append(ep)

    # ---- test ----
    model.eval()
    tst_preds, tst_labels = [], []
    with torch.no_grad():
        for batch in train_loader("test"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            tst_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tst_labels.extend(batch["label"].long().cpu().numpy())
    test_f1 = f1_score(tst_labels, tst_preds, average="macro")
    print(f"Test Macro-F1 (dropout={rate}): {test_f1:.4f}")

    rec["predictions"], rec["ground_truth"] = tst_preds, tst_labels
    rec["test_macro_f1"] = test_f1
    experiment_data["dropout_rate"][rate] = rec

    # plot per-rate loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(rec["epochs"], rec["losses"]["train"], label="train")
    plt.plot(rec["epochs"], rec["losses"]["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"Loss curve (dropout={rate})")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"loss_curve_dropout_{rate}.png"))
    plt.close()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
