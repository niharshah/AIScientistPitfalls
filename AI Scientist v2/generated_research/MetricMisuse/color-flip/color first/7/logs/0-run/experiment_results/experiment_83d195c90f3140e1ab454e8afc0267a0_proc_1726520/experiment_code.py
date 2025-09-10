import os, pathlib, random, time, json, math
from typing import List, Dict
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# experiment bookkeeping container
experiment_data = {
    "batch_size": {  # hyper-parameter tuned
        "SPR_BENCH": {
            # each batch size (32/64/128) will be added here
        }
    }
}

# ---------------------------------------------------------------------
# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------- Dataset utilities ----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_file):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    data = DatasetDict()
    data["train"] = _load("train.csv")
    data["dev"] = _load("dev.csv")
    data["test"] = _load("test.csv")
    return data


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
        print("SPR_BENCH not found – generating synthetic toy data.")
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
print(f"Classes={num_classes}  Train size={len(dset['train'])}")


# --------------------------- Vocab -----------------------------------
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1  # 0 reserved for padding
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
vocab_size = len(vocab) + 1
print("Vocab size:", vocab_size)


def encode_sequence(seq: str):
    return [vocab.get(tok, 0) for tok in seq.split()]


# --------------------- Metrics ---------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ------------------ Torch Dataset ------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = [encode_sequence(s) for s in hf_split["sequence"]]
        self.labels = hf_split["label"]
        self.raw = hf_split["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.raw[idx],
        }


def collate_fn(batch):
    lengths = [len(b["seq"]) for b in batch]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lengths[i]] = b["seq"]
    labels = torch.stack([b["label"] for b in batch])
    raw_seq = [b["raw_seq"] for b in batch]
    return {
        "seq": padded,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw_seq": raw_seq,
    }


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
        lens = mask.sum(1).clamp(min=1e-6)
        avg = summed / lens
        return self.fc(avg)


# ----------------- Training/eval routines ----------------------------
def evaluate(model, loader):
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
            raws.extend(batch["raw_seq"])
    return preds, labels, raws


def run_experiment(batch_size, epochs=5):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn
    )
    model = AvgEmbClassifier(vocab_size, 32, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    metrics_train, metrics_val = [], []
    losses_train, losses_val = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, n = 0.0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optim.zero_grad()
            logits = model(batch["seq"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optim.step()
            bs = batch["label"].size(0)
            tot_loss += loss.item() * bs
            n += bs
        train_loss = tot_loss / n

        # evaluation
        tr_preds, tr_lbls, tr_raw = evaluate(model, train_loader)
        dv_preds, dv_lbls, dv_raw = evaluate(model, dev_loader)

        tr_cwa = color_weighted_accuracy(tr_raw, tr_lbls, tr_preds)
        dv_cwa = color_weighted_accuracy(dv_raw, dv_lbls, dv_preds)
        tr_swa = shape_weighted_accuracy(tr_raw, tr_lbls, tr_preds)
        dv_swa = shape_weighted_accuracy(dv_raw, dv_lbls, dv_preds)
        tr_cpx = complexity_weighted_accuracy(tr_raw, tr_lbls, tr_preds)
        dv_cpx = complexity_weighted_accuracy(dv_raw, dv_lbls, dv_preds)

        metrics_train.append({"cwa": tr_cwa, "swa": tr_swa, "cpx": tr_cpx})
        metrics_val.append({"cwa": dv_cwa, "swa": dv_swa, "cpx": dv_cpx})
        losses_train.append(train_loss)
        losses_val.append(None)
        print(
            f"[bs={batch_size}] Epoch {ep}: "
            f"train_loss={train_loss:.4f}  Val CpxWA={dv_cpx:.4f}"
        )

    return {
        "metrics": {"train": metrics_train, "val": metrics_val},
        "losses": {"train": losses_train, "val": losses_val},
        "epochs": list(range(1, epochs + 1)),
    }


# ------------------ Hyper-parameter sweep ----------------------------
for bs in [32, 64, 128]:
    result = run_experiment(bs)
    experiment_data["batch_size"]["SPR_BENCH"][bs] = result

# ------------------ Save & plot --------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot validation CpxWA curves
plt.figure()
for bs, res in experiment_data["batch_size"]["SPR_BENCH"].items():
    cpx = [m["cpx"] for m in res["metrics"]["val"]]
    plt.plot(res["epochs"], cpx, marker="o", label=f"bs={bs}")
plt.title("Validation Complexity-Weighted Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("CpxWA")
plt.legend()
plt.savefig(os.path.join(working_dir, "cpxwa_curves.png"))
print("Finished. All results saved to 'working/'.")
