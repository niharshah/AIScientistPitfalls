import os, pathlib, random, time, json, math, warnings
from typing import List, Dict
import numpy as np
import torch, matplotlib

matplotlib.use("Agg")
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# ---------- Dataset loading helpers (adapted from SPR.py) ------------
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
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


# ---------------------------------------------------------------------
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
                "dev": load_dataset("json", data_files={"dev": dev}, split="dev"),
                "test": load_dataset("json", data_files={"test": test}, split="test"),
            }
        )
    return dset


dset = load_data()
num_classes = len(set(dset["train"]["label"]))
print(f"Classes: {num_classes}, Train size: {len(dset['train'])}")


# ---------------------------------------------------------------------
# --------------------- Tokenisation & Vocab --------------------------
def build_vocab(sequences: List[str]) -> Dict[str, int]:
    vocab = {}
    for seq in sequences:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1  # reserve 0 for padding
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
vocab_size = len(vocab) + 1
print(f"Vocab size: {vocab_size}")


def encode_sequence(sequence: str) -> List[int]:
    return [vocab.get(tok, 0) for tok in sequence.split()]


# ---------------------------------------------------------------------
# -------------------- Torch Dataset wrapper --------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.ids = hf_split["id"]
        self.seqs = [encode_sequence(s) for s in hf_split["sequence"]]
        self.labels = hf_split["label"]
        self.rawseq = hf_split["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.rawseq[idx],
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


train_ds = SPRTorchDataset(dset["train"])
dev_ds = SPRTorchDataset(dset["dev"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)


# ---------------------------------------------------------------------
# -------------------------- Model ------------------------------------
class AvgEmbClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int):
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


# ---------------------------------------------------------------------
# -------------- Experiment data tracking structure -------------------
experiment_data = {"emb_dim": {}}  # top-level key is tuning type

# ---------------------------------------------------------------------
# ------------------------- Training loop -----------------------------
EPOCHS = 5
emb_grid = [16, 32, 64, 128]

for emb_dim in emb_grid:
    print(f"\n===== Training with emb_dim={emb_dim} =====")
    model = AvgEmbClassifier(vocab_size, emb_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    # lists of raw sequences for metric computation
    train_raw = dset["train"]["sequence"]
    val_raw = dset["dev"]["sequence"]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
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
            total_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
        train_loss = total_loss / n

        # evaluation helper
        def evaluate(loader):
            model.eval()
            preds, labs, raws = [], [], []
            with torch.no_grad():
                for b in loader:
                    raw_seqs = b["raw"]
                    b = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in b.items()
                    }
                    logits = model(b["seq"])
                    preds.extend(logits.argmax(1).cpu().tolist())
                    labs.extend(b["label"].cpu().tolist())
                    raws.extend(raw_seqs)
            return preds, labs, raws

        train_preds, train_labels, _ = evaluate(train_loader)
        val_preds, val_labels, _ = evaluate(dev_loader)

        # compute metrics
        train_cwa = color_weighted_accuracy(train_raw, train_labels, train_preds)
        val_cwa = color_weighted_accuracy(val_raw, val_labels, val_preds)
        train_swa = shape_weighted_accuracy(train_raw, train_labels, train_preds)
        val_swa = shape_weighted_accuracy(val_raw, val_labels, val_preds)
        train_cpx = complexity_weighted_accuracy(train_raw, train_labels, train_preds)
        val_cpx = complexity_weighted_accuracy(val_raw, val_labels, val_preds)

        # logging
        log["metrics"]["train"].append(
            {"cwa": train_cwa, "swa": train_swa, "cpx": train_cpx}
        )
        log["metrics"]["val"].append({"cwa": val_cwa, "swa": val_swa, "cpx": val_cpx})
        log["losses"]["train"].append(train_loss)
        log["losses"]["val"].append(None)
        log["epochs"].append(epoch)

        print(
            f"[emb {emb_dim}] Epoch {epoch} | train_loss={train_loss:.4f} | Val CpxWA={val_cpx:.4f}"
        )

    # store last epoch predictions / ground truth
    log["predictions"] = val_preds
    log["ground_truth"] = val_labels

    experiment_data["emb_dim"][str(emb_dim)] = {"SPR_BENCH": log}

    # plot curve for this emb_dim
    cpx_vals = [m["cpx"] for m in log["metrics"]["val"]]
    plt.figure()
    plt.plot(log["epochs"], cpx_vals, marker="o")
    plt.title(f"Val Complexity-WA (emb_dim={emb_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.savefig(os.path.join(working_dir, f"cpxwa_curve_emb{emb_dim}.png"))
    plt.close()

    # free memory
    del model, optimizer, criterion
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------
# -------------------- Save metrics & plot ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiments finished. Data saved to working/experiment_data.npy")
