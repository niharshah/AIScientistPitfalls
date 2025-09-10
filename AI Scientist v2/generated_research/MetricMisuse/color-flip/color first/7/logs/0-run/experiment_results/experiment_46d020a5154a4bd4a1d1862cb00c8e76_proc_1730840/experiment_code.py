import os, pathlib, random, time, json
from typing import List, Dict
import numpy as np
import torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# ---------------------------- set-up ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------- data helpers ------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
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


def make_synth_split(n: int, vocab_shapes=5, vocab_colors=4, max_len=8, num_labels=3):
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
    except:
        print("SPR_BENCH not found – using synthetic data.")
        train, dev, test = [make_synth_split(n) for n in (3000, 600, 600)]
        return DatasetDict(
            {
                "train": load_dataset(
                    "json", data_files={"train": train}, split="train"
                ),
                "dev": load_dataset("json", data_files={"dev": dev}, split="dev"),
                "test": load_dataset("json", data_files={"test": test}, split="test"),
            }
        )


dset = load_data()
num_classes = len(set(dset["train"]["label"]))
print(f"Classes: {num_classes}, Train size: {len(dset['train'])}")


# ----------------------- metric functions ----------------------------
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def _wa(seqs, y_true, y_pred, weight_fn):
    w = [weight_fn(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(s, y_t, y_p):
    return _wa(s, y_t, y_p, count_color_variety)


def shape_weighted_accuracy(s, y_t, y_p):
    return _wa(s, y_t, y_p, count_shape_variety)


def complexity_weighted_accuracy(s, y_t, y_p):
    return _wa(s, y_t, y_p, lambda x: count_shape_variety(x) * count_color_variety(x))


# ---------------------- vocab & encoding -----------------------------
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1  # 0 reserved
    return vocab


vocab = build_vocab(dset["train"]["sequence"])
vocab_size = len(vocab) + 1
print(f"Vocab size: {vocab_size}")


def encode_sequence(seq: str) -> List[int]:
    return [vocab.get(tok, 0) for tok in seq.split()]


# --------------------- torch dataset --------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs_raw = hf_split["sequence"]
        self.seqs = [encode_sequence(s) for s in self.seqs_raw]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs_raw[idx],
        }


def collate_fn(batch):
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


train_ds, dev_ds = SPRTorchDataset(dset["train"]), SPRTorchDataset(dset["dev"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)


# ------------------------- model ------------------------------------
class AvgEmbClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        summed = (self.embed(x) * mask).sum(1)
        avg = summed / (mask.sum(1).clamp(min=1e-6))
        return self.fc(avg)


# -------------------- experiment tracking dict ----------------------
experiment_data = {"embedding_dim": {}}  # will fill per dimension


# ------------------- training & evaluation loop ---------------------
def run_training(emb_dim, epochs=5):
    model = AvgEmbClassifier(vocab_size, emb_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    run_log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": list(range(1, epochs + 1)),
    }
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0
        n = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["seq"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            opt.step()
            tot_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
        train_loss = tot_loss / n

        def evaluate(loader):
            model.eval()
            preds, labels, raws = [], [], []
            with torch.no_grad():
                for bt in loader:
                    bt = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in bt.items()
                    }
                    lg = model(bt["seq"])
                    preds.extend(lg.argmax(1).cpu().tolist())
                    labels.extend(bt["label"].cpu().tolist())
                    raws.extend(bt["raw"])
            return preds, labels, raws

        train_preds, train_labels, train_raw = evaluate(train_loader)
        val_preds, val_labels, val_raw = evaluate(dev_loader)

        # compute metrics
        train_cwa = color_weighted_accuracy(train_raw, train_labels, train_preds)
        val_cwa = color_weighted_accuracy(val_raw, val_labels, val_preds)
        train_swa = shape_weighted_accuracy(train_raw, train_labels, train_preds)
        val_swa = shape_weighted_accuracy(val_raw, val_labels, val_preds)
        train_cpx = complexity_weighted_accuracy(train_raw, train_labels, train_preds)
        val_cpx = complexity_weighted_accuracy(val_raw, val_labels, val_preds)

        run_log["metrics"]["train"].append(
            {"cwa": train_cwa, "swa": train_swa, "cpx": train_cpx}
        )
        run_log["metrics"]["val"].append(
            {"cwa": val_cwa, "swa": val_swa, "cpx": val_cpx}
        )
        run_log["losses"]["train"].append(train_loss)
        run_log["losses"]["val"].append(None)

        print(
            f"[emb={emb_dim}] Epoch {ep}/{epochs}  train_loss={train_loss:.4f}  Val CpxWA={val_cpx:.4f}"
        )
    run_log["predictions"] = val_preds
    run_log["ground_truth"] = val_labels
    return run_log


# ------------------- hyper-parameter sweep --------------------------
sweep_dims = [32, 64, 128, 256]
for dim in sweep_dims:
    experiment_data["embedding_dim"][f"dim_{dim}"] = run_training(dim, epochs=5)

# -------------- save experiment data & plots per run ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot comparison of validation CpxWA curves
plt.figure()
for dim in sweep_dims:
    vals = [
        m["cpx"]
        for m in experiment_data["embedding_dim"][f"dim_{dim}"]["metrics"]["val"]
    ]
    plt.plot(
        experiment_data["embedding_dim"][f"dim_{dim}"]["epochs"],
        vals,
        marker="o",
        label=f"dim{dim}",
    )
plt.title("Validation Complexity-Weighted Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("CpxWA")
plt.legend()
plt.savefig(os.path.join(working_dir, "cpxwa_dim_compare.png"))
print("Finished. Results stored in working/.")
