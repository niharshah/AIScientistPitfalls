import os, random, string, pathlib, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict

# --------------------------- experiment data dict ---------------------------
experiment_data = {
    "weight_decay": {  # <- hyperparam tuning type
        # each key added later: str(weight_decay)
    }
}

# --------------------------- working dir & device ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------- load or build dataset --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

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


def build_synthetic_dataset(n_train=2000, n_dev=500, n_test=500, max_len=12):
    def _row():
        length = random.randint(4, max_len)
        label, seq = 0, []
        for _ in range(length):
            shape, color = random.choice(string.ascii_uppercase[:5]), random.choice(
                "01234"
            )
            seq.append(shape + color)
            label ^= (ord(shape) + int(color)) & 1
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": int(label),
        }

    from datasets import Dataset

    return DatasetDict(
        {
            "train": Dataset.from_list([_row() for _ in range(n_train)]),
            "dev": Dataset.from_list([_row() for _ in range(n_dev)]),
            "test": Dataset.from_list([_row() for _ in range(n_test)]),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
spr = load_spr_bench(DATA_PATH) if DATA_PATH.exists() else build_synthetic_dataset()
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# --------------------------- vocabulary & encoding --------------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx, MAX_LEN = vocab[PAD], 40


def encode_sequence(seq, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# --------------------------- torch dataset ----------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return {
            "input_ids": torch.tensor(
                encode_sequence(row["sequence"]), dtype=torch.long
            ),
            "labels": torch.tensor(row["label"], dtype=torch.long),
            "sequence": row["sequence"],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "sequence": [b["sequence"] for b in batch],
    }


# --------------------------- model ------------------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)  # B,L,D
        mask = (x != pad_idx).unsqueeze(-1).float()  # B,L,1
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc(pooled)


# --------------------------- augmentation helper ----------------------------
def shape_rename(seq):
    toks = seq.split()
    shapes = list({t[0] for t in toks})
    mapping = {s: random.choice(string.ascii_uppercase) for s in shapes}
    new_toks = [mapping[t[0]] + t[1:] for t in toks]
    return " ".join(new_toks)


# --------------------------- ACS metric -------------------------------------
@torch.no_grad()
def compute_ACS(model, dataset, max_samples=500, n_aug=5):
    model.eval()
    total, consist = 0, 0.0
    for i in range(min(len(dataset), max_samples)):
        base = dataset[i]["sequence"]
        label = dataset[i]["label"]
        variants = [base] + [shape_rename(base) for _ in range(n_aug)]
        correct = sum(
            model(
                torch.tensor(
                    encode_sequence(v), dtype=torch.long, device=device
                ).unsqueeze(0)
            )
            .argmax(-1)
            .item()
            == label
            for v in variants
        )
        consist += correct / len(variants)
        total += 1
    return consist / total if total else 0.0


# --------------------------- training loop with grid search -----------------
weight_decays = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
EPOCHS = 5
batch_size_tr, batch_size_val = 128, 256
criterion = nn.CrossEntropyLoss()

for wd in weight_decays:
    print(f"\n=== Training with weight_decay = {wd} ===")
    key = str(wd)
    experiment_data["weight_decay"][key] = {
        "SPR_BENCH": {
            "metrics": {"train_loss": [], "val_loss": [], "val_ACS": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    # data loaders re-constructed each run (for shuffling independence)
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]),
        batch_size=batch_size_tr,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        SPRTorchDataset(spr["dev"]),
        batch_size=batch_size_val,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = MeanEmbedClassifier(len(vocab), 128, len(set(spr["train"]["label"]))).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=wd)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # on-the-fly 50% augmentation
            seq_aug = [
                shape_rename(s) if random.random() < 0.5 else s
                for s in batch["sequence"]
            ]
            batch["input_ids"] = torch.stack(
                [torch.tensor(encode_sequence(s), dtype=torch.long) for s in seq_aug]
            )
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        experiment_data["weight_decay"][key]["SPR_BENCH"]["metrics"][
            "train_loss"
        ].append((epoch, train_loss))

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                bt = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(bt["input_ids"])
                loss = criterion(logits, bt["labels"])
                val_loss += loss.item()
                experiment_data["weight_decay"][key]["SPR_BENCH"]["predictions"].extend(
                    logits.argmax(-1).cpu().tolist()
                )
                experiment_data["weight_decay"][key]["SPR_BENCH"][
                    "ground_truth"
                ].extend(batch["labels"].cpu().tolist())
        val_loss /= len(dev_loader)
        experiment_data["weight_decay"][key]["SPR_BENCH"]["metrics"]["val_loss"].append(
            (epoch, val_loss)
        )

        val_ACS = compute_ACS(model, spr["dev"])
        experiment_data["weight_decay"][key]["SPR_BENCH"]["metrics"]["val_ACS"].append(
            (epoch, val_ACS)
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ACS={val_ACS:.4f}"
        )

# --------------------------- save everything --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved metrics to", os.path.join(working_dir, "experiment_data.npy"))
