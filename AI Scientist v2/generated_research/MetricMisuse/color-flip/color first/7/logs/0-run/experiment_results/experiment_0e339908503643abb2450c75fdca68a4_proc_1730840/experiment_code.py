import os, pathlib, random, math, numpy as np, torch
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# -------------------- Dataset helpers --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


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


# ---------------- Synthetic fallback ---------------------------------
def make_synthetic_split(n, vocab_shapes=5, vocab_colors=4, max_len=8, num_labels=3):
    rng = random.Random(42 + n)
    data = {"id": [], "sequence": [], "label": []}
    for i in range(n):
        L = rng.randint(3, max_len)
        seq = " ".join(
            chr(ord("A") + rng.randint(0, vocab_shapes - 1))
            + str(rng.randint(0, vocab_colors - 1))
            for _ in range(L)
        )
        data["id"].append(str(i))
        data["sequence"].append(seq)
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
        train, dev, test = map(make_synthetic_split, (3000, 600, 600))
        return DatasetDict(
            {
                "train": load_dataset(
                    "json", data_files={"train": train}, split="train"
                ),
                "dev": load_dataset("json", data_files={"train": dev}, split="train"),
                "test": load_dataset("json", data_files={"train": test}, split="train"),
            }
        )


dset = load_data()
num_classes = len(set(dset["train"]["label"]))
print(f"Classes: {num_classes}, Train size: {len(dset['train'])}")


# --------------------- Tokenisation ----------------------------------
def build_vocab(sequences: List[str]) -> Dict[str, int]:
    vocab = {}
    for seq in sequences:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1  # 0 = padding
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
vocab_size = len(vocab) + 1
print(f"Vocab size: {vocab_size}")


def encode_sequence(sequence: str) -> List[int]:
    return [vocab.get(tok, 0) for tok in sequence.split()]


# -------------------- Torch Dataset ----------------------------------
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


train_ds, dev_ds = SPRTorchDataset(dset["train"]), SPRTorchDataset(dset["dev"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)


# -------------------------- Model ------------------------------------
class AvgEmbClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, num_cls)
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        summed = (self.embed(x) * mask).sum(1)
        avg = summed / mask.sum(1).clamp(min=1e-6)
        return self.fc(avg)


# ---------------------------------------------------------------------
# Hyper-parameter tuning: various LR schedulers
schedules = [
    {"name": "constant", "type": "none"},
    {"name": "cosine_warm1", "type": "cosine", "warmup_epochs": 1},
    {"name": "step_gamma05", "type": "step", "step_size": 2, "gamma": 0.5},
]

experiment_data = {"learning_rate_scheduler": {}}
criterion = nn.CrossEntropyLoss()
EPOCHS = 5


def evaluate(model, loader):
    model.eval()
    preds, labels, raws = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["seq"])
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
            raws.extend(batch["raw"])
    return preds, labels, raws


for cfg in schedules:
    print(f"\n=== Training with scheduler: {cfg['name']} ===")
    # fresh model / optim / scheduler
    model = AvgEmbClassifier(vocab_size, 32, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if cfg["type"] == "cosine":

        def lr_lambda(epoch):
            warm = cfg["warmup_epochs"]
            if epoch < warm:
                return (epoch + 1) / warm
            progress = (epoch - warm) / max(1, EPOCHS - warm)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg["type"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
        )
    else:
        scheduler = None

    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "lr": [],
        "epochs": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, n = 0.0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["seq"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
        # end epoch update
        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- evaluation
        train_preds, train_labels, train_raws = evaluate(model, train_loader)
        val_preds, val_labels, val_raws = evaluate(model, dev_loader)

        train_cpx = complexity_weighted_accuracy(train_raws, train_labels, train_preds)
        val_cpx = complexity_weighted_accuracy(val_raws, val_labels, val_preds)

        exp_rec["metrics"]["train"].append({"cpx": train_cpx})
        exp_rec["metrics"]["val"].append({"cpx": val_cpx})
        exp_rec["losses"]["train"].append(loss.item())
        exp_rec["losses"]["val"].append(None)
        exp_rec["lr"].append(current_lr)
        exp_rec["epochs"].append(epoch)
        print(f"Epoch {epoch} | LR {current_lr:.5f} | Val CpxWA {val_cpx:.4f}")

    experiment_data["learning_rate_scheduler"][cfg["name"]] = exp_rec

# ---------------------------------------------------------------------
# save & quick plot of best schedule
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

for name, rec in experiment_data["learning_rate_scheduler"].items():
    plt.plot(
        rec["epochs"], [m["cpx"] for m in rec["metrics"]["val"]], marker="o", label=name
    )
plt.title("Validation Complexity-Weighted Accuracy")
plt.xlabel("Epoch")
plt.ylabel("CpxWA")
plt.legend()
plt.savefig(os.path.join(working_dir, "cpxwa_curve.png"))
print("Finished. Data saved to working/experiment_data.npy")
