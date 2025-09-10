import os, pathlib, random, time, json, math
from typing import List, Dict
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------- experiment store --------------------------
experiment_data = {
    "epochs_tuning": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
exp_ref = experiment_data["epochs_tuning"]["SPR_BENCH"]

# ------------------------- working dir -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------- device --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- Dataset helpers ------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# --------------------- synthetic fallback ----------------------------
def make_synthetic_split(n, vocab_shapes=5, vocab_colors=4, max_len=8, num_labels=3):
    rng = random.Random(42 + n)
    data = {"id": [], "sequence": [], "label": []}
    for i in range(n):
        L = rng.randint(3, max_len)
        seq = [
            chr(ord("A") + rng.randint(0, vocab_shapes - 1))
            + str(rng.randint(0, vocab_colors - 1))
            for _ in range(L)
        ]
        data["id"].append(str(i))
        data["sequence"].append(" ".join(seq))
        data["label"].append(rng.randint(0, num_labels - 1))
    return data


def load_data():
    spr_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        if not spr_root.exists():
            raise FileNotFoundError
        return load_spr_bench(spr_root)
    except Exception:
        print("SPR_BENCH not found – using synthetic data.")
        return DatasetDict(
            {
                "train": load_dataset(
                    "json",
                    data_files={"train": make_synthetic_split(3000)},
                    split="train",
                ),
                "dev": load_dataset(
                    "json",
                    data_files={"train": make_synthetic_split(600)},
                    split="train",
                ),
                "test": load_dataset(
                    "json",
                    data_files={"train": make_synthetic_split(600)},
                    split="train",
                ),
            }
        )


dset = load_data()
num_classes = len(set(dset["train"]["label"]))
print(f'Classes:{num_classes}, Train size:{len(dset["train"])}')


# ------------------------ tokenisation -------------------------------
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
vocab_size = len(vocab) + 1


def encode_sequence(seq: str) -> List[int]:
    return [vocab.get(tok, 0) for tok in seq.split()]


# ------------------------ torch dataset ------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, raw_sequences):
        self.seqs = [encode_sequence(s) for s in hf_split["sequence"]]
        self.labels = hf_split["label"]
        self.raw = raw_sequences

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.raw[idx],
        }


def collate_fn(batch):
    lengths = [len(b["seq"]) for b in batch]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lengths[i]] = b["seq"]
    labels = torch.stack([b["label"] for b in batch])
    raws = [b["raw"] for b in batch]
    return {
        "seq": padded,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw": raws,
    }


train_ds = SPRTorchDataset(dset["train"], dset["train"]["sequence"])
dev_ds = SPRTorchDataset(dset["dev"], dset["dev"]["sequence"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)


# ---------------------------- model ----------------------------------
class AvgEmbClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        summed = (self.embed(x) * mask).sum(1)
        avg = summed / mask.sum(1).clamp(min=1e-6)
        return self.fc(avg)


model = AvgEmbClassifier(vocab_size, 32, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------------- training --------------------------------
MAX_EPOCHS = 30
patience = 5
best_val = -float("inf")
patience_ctr = 0


def run_eval(loader):
    model.eval()
    preds, labels, raws = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["seq"])
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
            raws.extend(batch["raw"])
    return preds, labels, raws


for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tot_loss = 0
    n = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["seq"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["label"].size(0)
        n += batch["label"].size(0)
    train_loss = tot_loss / n

    # evaluation
    tr_preds, tr_labels, _ = run_eval(train_loader)
    val_preds, val_labels, val_raw = run_eval(dev_loader)
    train_raw = dset["train"]["sequence"]

    train_cwa = color_weighted_accuracy(train_raw, tr_labels, tr_preds)
    val_cwa = color_weighted_accuracy(val_raw, val_labels, val_preds)
    train_swa = shape_weighted_accuracy(train_raw, tr_labels, tr_preds)
    val_swa = shape_weighted_accuracy(val_raw, val_labels, val_preds)
    train_cpx = complexity_weighted_accuracy(train_raw, tr_labels, tr_preds)
    val_cpx = complexity_weighted_accuracy(val_raw, val_labels, val_preds)

    # logging
    exp_ref["metrics"]["train"].append(
        {"cwa": train_cwa, "swa": train_swa, "cpx": train_cpx}
    )
    exp_ref["metrics"]["val"].append({"cwa": val_cwa, "swa": val_swa, "cpx": val_cpx})
    exp_ref["losses"]["train"].append(train_loss)
    exp_ref["losses"]["val"].append(None)
    exp_ref["predictions"].append(val_preds)
    exp_ref["ground_truth"].append(val_labels)
    exp_ref["epochs"].append(epoch)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS}  train_loss={train_loss:.4f}  Val CpxWA={val_cpx:.4f}"
    )

    # early stopping
    if val_cpx > best_val + 1e-6:
        best_val = val_cpx
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print(f"No improvement for {patience} epochs – early stopping.")
            break

# -------------------------- save & plot ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

cpx_vals = [m["cpx"] for m in exp_ref["metrics"]["val"]]
plt.figure()
plt.plot(exp_ref["epochs"], cpx_vals, marker="o")
plt.title("Validation Complexity-Weighted Accuracy")
plt.xlabel("Epoch")
plt.ylabel("CpxWA")
plt.savefig(os.path.join(working_dir, "cpxwa_curve.png"))
print("Finished. All artifacts saved in working/.")
