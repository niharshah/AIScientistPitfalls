# --------------------------------------------------
# Hyper-parameter sweep : aug_probability
# --------------------------------------------------
import os, random, string, pathlib, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper : load SPR-BENCH (or synthetic) ----------
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


def build_synthetic_dataset(n_train=2000, n_dev=500, n_test=500, max_len=12):
    def _gen_row():
        length = random.randint(4, max_len)
        seq, label = [], 0
        for _ in range(length):
            shape = random.choice(string.ascii_uppercase[:5])  # 5 shapes
            color = random.choice("01234")  # 5 colors
            seq.append(shape + color)
            label ^= (ord(shape) + int(color)) & 1  # parity rule
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": int(label),
        }

    def _many(n):
        return [_gen_row() for _ in range(n)]

    from datasets import Dataset

    return DatasetDict(
        {
            "train": Dataset.from_list(_many(n_train)),
            "dev": Dataset.from_list(_many(n_dev)),
            "test": Dataset.from_list(_many(n_test)),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH from disk.")
else:
    spr = build_synthetic_dataset()
    print("SPR_BENCH folder not found, using synthetic data.")

# ---------- vocabulary ----------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]

# ---------- encode ----------
MAX_LEN = 40


def encode_sequence(seq, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
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


train_loader_base = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        mask = (x != pad_idx).unsqueeze(-1).float()
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc(pooled)


# ---------- augmentation ----------
def shape_rename(seq):
    toks = seq.strip().split()
    shapes = list({t[0] for t in toks})
    mapping = {s: random.choice(string.ascii_uppercase) for s in shapes}
    new_toks = [mapping[t[0]] + t[1:] if len(t) > 1 else mapping[t[0]] for t in toks]
    return " ".join(new_toks)


# ---------- ACS metric ----------
@torch.no_grad()
def compute_ACS(model, dataset, max_samples=500, n_aug=5):
    model.eval()
    total, consist = 0, 0.0
    for i in range(min(len(dataset), max_samples)):
        row = dataset[i]
        variants = [row["sequence"]] + [
            shape_rename(row["sequence"]) for _ in range(n_aug)
        ]
        correct = 0
        for sv in variants:
            ids = (
                torch.tensor(encode_sequence(sv), dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )
            if model(ids).argmax(-1).item() == row["label"]:
                correct += 1
        consist += correct / len(variants)
        total += 1
    return consist / total if total else 0.0


# ---------- experiment data ----------
experiment_data = {"aug_probability": {"SPR_BENCH": {}}}

# ---------- training & sweep ----------
prob_list = [0.0, 0.25, 0.5, 0.75, 1.0]
EPOCHS = 5
num_labels = len(set(spr["train"]["label"]))

for prob in prob_list:
    print(f"\n=== Training with aug_probability = {prob} ===")
    model = MeanEmbedClassifier(len(vocab), 128, num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    # record containers
    run_dict = {
        "train_loss": [],
        "val_loss": [],
        "val_ACS": [],
        "predictions": [],
        "ground_truth": [],
    }

    # (need independent train loader each run because shuffle state)
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            # apply augmentation with given probability
            seqs_aug = [
                shape_rename(s) if random.random() < prob else s
                for s in batch["sequence"]
            ]
            batch["input_ids"] = torch.stack(
                [torch.tensor(encode_sequence(s), dtype=torch.long) for s in seqs_aug]
            )
            # move to device
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch_t["input_ids"])
            loss = criterion(logits, batch_t["labels"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        train_loss = tot_loss / len(train_loader)

        # validation
        model.eval()
        val_loss, preds, gts = 0.0, [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch_t = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch_t["input_ids"])
                val_loss += criterion(logits, batch_t["labels"]).item()
                preds.extend(logits.argmax(-1).cpu().tolist())
                gts.extend(batch["labels"].cpu().tolist())
        val_loss /= len(dev_loader)
        val_ACS = compute_ACS(model, spr["dev"])

        run_dict["train_loss"].append((epoch, train_loss))
        run_dict["val_loss"].append((epoch, val_loss))
        run_dict["val_ACS"].append((epoch, val_ACS))

        print(
            f"Epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | ACS {val_ACS:.4f}"
        )

    run_dict["predictions"] = preds
    run_dict["ground_truth"] = gts
    experiment_data["aug_probability"]["SPR_BENCH"][prob] = run_dict

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy with all sweep results.")
