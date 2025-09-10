#!/usr/bin/env python
# Hyper-parameter tuning: batch_size
import os, pathlib, random, time, json, math, warnings
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# directory & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# ---------- Dataset loading helpers ----------------------------------
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


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------------- Synthetic fallback -----------------------------
def make_synthetic_split(
    n: int, vocab_shapes=5, vocab_colors=4, max_len=8, num_labels=3
):
    rng = random.Random(42 + n)
    data = {"id": [], "sequence": [], "label": []}
    for i in range(n):
        L = rng.randint(3, max_len)
        seq = []
        for _ in range(L):
            s = chr(ord("A") + rng.randint(0, vocab_shapes - 1))
            c = str(rng.randint(0, vocab_colors - 1))
            seq.append(s + c)
        data["id"].append(str(i))
        data["sequence"].append(" ".join(seq))
        data["label"].append(rng.randint(0, num_labels - 1))
    return data


def load_data():
    spr_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        if not spr_root.exists():
            raise FileNotFoundError
        dset = load_spr_bench(spr_root)
    except Exception:
        print("SPR_BENCH not found – using synthetic data.")
        train = make_synthetic_split(3000)
        dev = make_synthetic_split(600)
        test = make_synthetic_split(600)
        dset = DatasetDict(
            {
                "train": load_dataset(
                    "json", data_files={"train": train}, split="train"
                ),
                "dev": load_dataset("json", data_files={"train": dev}, split="train"),
                "test": load_dataset("json", data_files={"train": test}, split="train"),
            }
        )
    return dset


dset = load_data()
num_classes = len(set(dset["train"]["label"]))
print(f"Classes: {num_classes}, Train size: {len(dset['train'])}")


# --------------------- Tokenisation & Vocab --------------------------
def build_vocab(sequences: List[str]) -> Dict[str, int]:
    vocab = {}
    for seq in sequences:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1  # 0 reserved for pad
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
vocab_size = len(vocab) + 1
print(f"Vocab size: {vocab_size}")


def encode_sequence(sequence: str) -> List[int]:
    return [vocab.get(tok, 0) for tok in sequence.split()]


# -------------------- Torch Dataset wrapper --------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = [encode_sequence(s) for s in hf_split["sequence"]]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    lengths = [len(b["seq"]) for b in batch]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lengths[i]] = b["seq"]
    labels = torch.stack([b["label"] for b in batch])
    return {"seq": padded, "lengths": torch.tensor(lengths), "label": labels}


train_ds = SPRTorchDataset(dset["train"])
dev_ds = SPRTorchDataset(dset["dev"])


# -------------------------- Model ------------------------------------
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
        lengths = mask.sum(1).clamp(min=1e-6)
        avg = summed / lengths
        return self.fc(avg)


# ---------------- Experiment data structure --------------------------
experiment_data = {
    "batch_size_tuning": {
        "SPR_BENCH": {
            "batch_sizes": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
        }
    }
}

# ------------------------- Training loop -----------------------------
EPOCHS = 5
candidate_batch_sizes = [32, 64, 128, 256]

for bs in candidate_batch_sizes:
    print(f"\n===== Training with batch_size={bs} =====")
    # data loaders
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=max(256, bs), shuffle=False, collate_fn=collate_fn
    )

    # model/optim
    model = AvgEmbClassifier(vocab_size, 32, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # containers for this batch_size
    train_metrics_epochs, val_metrics_epochs = [], []
    train_losses_epochs, val_losses_epochs = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["seq"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
        train_loss = tot_loss / n

        # ----- Evaluation
        def get_preds(loader):
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for b in loader:
                    b = {k: v.to(device) for k, v in b.items()}
                    logits = model(b["seq"])
                    preds.extend(logits.argmax(1).cpu().tolist())
                    labels.extend(b["label"].cpu().tolist())
            return preds, labels

        train_preds, train_labels = get_preds(train_loader)
        val_preds, val_labels = get_preds(dev_loader)

        train_raw = dset["train"]["sequence"]
        val_raw = dset["dev"]["sequence"]

        # metrics
        tm = {
            "cwa": color_weighted_accuracy(train_raw, train_labels, train_preds),
            "swa": shape_weighted_accuracy(train_raw, train_labels, train_preds),
            "cpx": complexity_weighted_accuracy(train_raw, train_labels, train_preds),
        }
        vm = {
            "cwa": color_weighted_accuracy(val_raw, val_labels, val_preds),
            "swa": shape_weighted_accuracy(val_raw, val_labels, val_preds),
            "cpx": complexity_weighted_accuracy(val_raw, val_labels, val_preds),
        }

        train_metrics_epochs.append(tm)
        val_metrics_epochs.append(vm)
        train_losses_epochs.append(train_loss)
        val_losses_epochs.append(None)  # val loss not computed
        print(
            f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | Val CpxWA={vm['cpx']:.4f}"
        )

    # store
    exp_ds = experiment_data["batch_size_tuning"]["SPR_BENCH"]
    exp_ds["batch_sizes"].append(bs)
    exp_ds["metrics"]["train"].append(train_metrics_epochs)
    exp_ds["metrics"]["val"].append(val_metrics_epochs)
    exp_ds["losses"]["train"].append(train_losses_epochs)
    exp_ds["losses"]["val"].append(val_losses_epochs)
    exp_ds["epochs"].append(list(range(1, EPOCHS + 1)))

# ---------------------------------------------------------------------
# -------------------- Save metrics & plot ----------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# optional plot for quick visual
plt.figure(figsize=(6, 4))
for idx, bs in enumerate(
    experiment_data["batch_size_tuning"]["SPR_BENCH"]["batch_sizes"]
):
    val_cpx = [
        e["cpx"]
        for e in experiment_data["batch_size_tuning"]["SPR_BENCH"]["metrics"]["val"][
            idx
        ]
    ]
    plt.plot(range(1, EPOCHS + 1), val_cpx, marker="o", label=f"bs={bs}")
plt.title("Validation Complexity-Weighted Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("CpxWA")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cpxwa_batchsize_curve.png"))
print("Finished. Results saved in working/.")
