import os, pathlib, random, time, json
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------
# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------------------------------------
# helpers for SPR_BENCH -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def color_weighted_accuracy(seqs, y, yhat):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y, yhat) if t == p) / max(1, sum(w))


def shape_weighted_accuracy(seqs, y, yhat):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y, yhat) if t == p) / max(1, sum(w))


def complexity_weighted_accuracy(seqs, y, yhat):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y, yhat) if t == p) / max(1, sum(w))


# synthetic fallback ----------------------------------------------------
def make_synthetic_split(n, vocab_shapes=5, vocab_colors=4, max_len=8, num_labels=3):
    rng = random.Random(1337 + n)
    data = []
    for i in range(n):
        L = rng.randint(3, max_len)
        seq = []
        for _ in range(L):
            s = chr(ord("A") + rng.randint(0, vocab_shapes - 1))
            c = str(rng.randint(0, vocab_colors - 1))
            seq.append(s + c)
        data.append(
            {
                "id": str(i),
                "sequence": " ".join(seq),
                "label": rng.randint(0, num_labels - 1),
            }
        )
    return data


def load_data():
    spr_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        if not spr_root.exists():
            raise FileNotFoundError
        return load_spr_bench(spr_root)
    except Exception:
        print("SPR_BENCH not found – generating synthetic data.")
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
print(f"Train size: {len(dset['train'])}, classes: {num_classes}")


# ------------------ vocabulary ----------------------------------------
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1  # 0==pad
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
vocab_size = len(vocab) + 1
print("Vocab size:", vocab_size)


def encode(seq: str):
    return [vocab.get(tok, 0) for tok in seq.split()]


# ------------------ torch dataset -------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.sequences = hf_split["sequence"]
        self.enc = [encode(s) for s in self.sequences]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.enc[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.sequences[idx],
        }


def collate(batch):
    lengths = [len(b["seq"]) for b in batch]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lengths[i]] = b["seq"]
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "seq": padded,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw": raw,
    }


train_loader = DataLoader(
    SPRTorchDataset(dset["train"]), batch_size=64, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset(dset["dev"]), batch_size=128, shuffle=False, collate_fn=collate
)


# ------------------ model ---------------------------------------------
class AvgEmbClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        summed = (self.embed(x) * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1e-6)
        avg = summed / lengths
        return self.fc(avg)


# ------------------ experiment tracking dict --------------------------
experiment_data = {"fc_hidden_dim": {"SPR_BENCH": {}}}

# ------------------ hyperparameter loop -------------------------------
hidden_dims = [64, 128, 256, 512]
EPOCHS = 5
for hdim in hidden_dims:
    print(f"\n=== Training with hidden_dim={hdim} ===")
    data_store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    model = AvgEmbClassifier(vocab_size, 32, hdim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # training epochs
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0
        n = 0
        for batch in train_loader:
            for k in ["seq", "label"]:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            logits = model(batch["seq"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
        train_loss = running_loss / n

        # evaluation ----------------------------------------------------
        def get_preds(loader):
            model.eval()
            preds, labels, raws = [], [], []
            with torch.no_grad():
                for b in loader:
                    lbl = b["label"].to(device)
                    out = model(b["seq"].to(device))
                    preds.extend(out.argmax(1).cpu().tolist())
                    labels.extend(b["label"].tolist())
                    raws.extend(b["raw"])
            return preds, labels, raws

        tr_preds, tr_labels, tr_raw = get_preds(train_loader)
        dv_preds, dv_labels, dv_raw = get_preds(dev_loader)
        # metrics
        tr_cwa = color_weighted_accuracy(tr_raw, tr_labels, tr_preds)
        dv_cwa = color_weighted_accuracy(dv_raw, dv_labels, dv_preds)
        tr_swa = shape_weighted_accuracy(tr_raw, tr_labels, tr_preds)
        dv_swa = shape_weighted_accuracy(dv_raw, dv_labels, dv_preds)
        tr_cpx = complexity_weighted_accuracy(tr_raw, tr_labels, tr_preds)
        dv_cpx = complexity_weighted_accuracy(dv_raw, dv_labels, dv_preds)
        # store
        data_store["metrics"]["train"].append(
            {"cwa": tr_cwa, "swa": tr_swa, "cpx": tr_cpx}
        )
        data_store["metrics"]["val"].append(
            {"cwa": dv_cwa, "swa": dv_swa, "cpx": dv_cpx}
        )
        data_store["losses"]["train"].append(train_loss)
        data_store["losses"]["val"].append(None)
        data_store["epochs"].append(epoch)
        print(f"Epoch {epoch}: loss={train_loss:.4f}  Val CpxWA={dv_cpx:.4f}")
    # save curves for this hyperparam ----------------------------------
    plt.figure()
    vals = [m["cpx"] for m in data_store["metrics"]["val"]]
    plt.plot(data_store["epochs"], vals, marker="o")
    plt.title(f"Val CpxWA (hidden={hdim})")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.savefig(os.path.join(working_dir, f"cpxwa_h{hdim}.png"))
    plt.close()
    experiment_data["fc_hidden_dim"]["SPR_BENCH"][str(hdim)] = data_store

# ------------------ persist -------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. All data saved to working/.")
