# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, string, pathlib, math, time, json
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset as HFDataset, DatasetDict, load_dataset

# ------------------------------ house-keeping ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ------------------------------ data utils ---------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


def build_synthetic_dataset(n_train=2000, n_dev=500, n_test=500, max_len=12):
    def _gen_row():
        l = random.randint(4, max_len)
        seq, label = [], 0
        for _ in range(l):
            sh, co = random.choice(string.ascii_uppercase[:5]), random.choice("01234")
            seq.append(sh + co)
            label ^= (ord(sh) + int(co)) & 1
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": label,
        }

    def _many(n):
        return [_gen_row() for _ in range(n)]

    return DatasetDict(
        train=HFDataset.from_list(_many(n_train)),
        dev=HFDataset.from_list(_many(n_dev)),
        test=HFDataset.from_list(_many(n_test)),
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_synthetic_dataset()
print("Dataset size:", {k: len(v) for k, v in spr.items()})

# ------------------------------ vocabulary ---------------------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]

MAX_LEN = 40


def encode_sequence(seq, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# ------------------------------ metrics ------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ------------------------------ dataset wrappers ---------------------------
class SPRTorchDataset(TorchDataset):
    def __init__(self, hf_dataset: HFDataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        return {
            "sequence": row["sequence"],
            "input_ids": torch.tensor(
                encode_sequence(row["sequence"]), dtype=torch.long
            ),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "sequence": [b["sequence"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def shape_rename(seq):
    toks = seq.split()
    shapes = list({t[0] for t in toks})
    mapping = {s: random.choice(string.ascii_uppercase) for s in shapes}
    return " ".join([mapping[t[0]] + t[1:] for t in toks])


# ------------------------------ model --------------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_cls)

    def forward(self, x):
        emb = self.embed(x)  # B,L,D
        mask = (x != pad_idx).unsqueeze(-1).float()  # B,L,1
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc(pooled)


# ------------------------------ experiment store ---------------------------
experiment_data = {"embed_dim_tuning": {}}

# ------------------------------ sweep --------------------------------------
embed_dims = [64, 128, 256]
EPOCHS = 5

for dim in embed_dims:
    print(f"\n=== Training embed_dim={dim} ===")
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        SPRTorchDataset(spr["dev"]),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = MeanEmbedClassifier(len(vocab), dim, len(set(spr["train"]["label"]))).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()

    run_store = {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "CoWA": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # -------------------- train --------------------
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            # 50 % augmentation
            seq_aug = [
                shape_rename(s) if random.random() < 0.5 else s
                for s in batch["sequence"]
            ]
            batch["input_ids"] = torch.stack(
                [torch.tensor(encode_sequence(s), dtype=torch.long) for s in seq_aug]
            )
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["labels"])
            loss.backward()
            opt.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        run_store["losses"]["train"].append((epoch, train_loss))

        # -------------------- validation ----------------
        model.eval()
        val_loss, seqs, gts, preds = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch_cuda = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch_cuda["input_ids"])
                val_loss += crit(logits, batch_cuda["labels"]).item()
                p = logits.argmax(-1).cpu().tolist()
                preds.extend(p)
                gts.extend(batch["labels"].tolist())
                seqs.extend(batch["sequence"])
        val_loss /= len(dev_loader)
        SWA = shape_weighted_accuracy(seqs, gts, preds)
        CWA = color_weighted_accuracy(seqs, gts, preds)
        CoWA = (SWA + CWA) / 2

        run_store["losses"]["val"].append((epoch, val_loss))
        run_store["metrics"]["SWA"].append((epoch, SWA))
        run_store["metrics"]["CWA"].append((epoch, CWA))
        run_store["metrics"]["CoWA"].append((epoch, CoWA))
        run_store["predictions"].append((epoch, preds))
        run_store["ground_truth"].append((epoch, gts))

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={SWA:.4f} CWA={CWA:.4f} CoWA={CoWA:.4f}"
        )

    experiment_data["embed_dim_tuning"][f"embed_{dim}"] = run_store

# ------------------------------ save ---------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
