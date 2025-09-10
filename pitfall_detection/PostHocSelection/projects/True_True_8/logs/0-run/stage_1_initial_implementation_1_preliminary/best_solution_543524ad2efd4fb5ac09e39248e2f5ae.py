import os, random, string, pathlib, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper : load dataset (provided util) ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset, DatasetDict

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


# ---------- synthetic fallback ----------
def build_synthetic_dataset(n_train=2000, n_dev=500, n_test=500, max_len=12):
    def _gen_row():
        length = random.randint(4, max_len)
        seq = []
        label = 0
        for _ in range(length):
            shape = random.choice(string.ascii_uppercase[:5])  # 5 shapes
            color = random.choice("01234")  # 5 colors
            seq.append(shape + color)
            label ^= (ord(shape) + int(color)) & 1  # synthetic rule: parity
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": int(label),
        }

    def _many(n):
        return [_gen_row() for _ in range(n)]

    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_list(_many(n_train)),
            "dev": Dataset.from_list(_many(n_dev)),
            "test": Dataset.from_list(_many(n_test)),
        }
    )


# ---------- dataset loading ----------
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


# ---------- encode function ----------
def encode_sequence(seq, max_len=40):
    toks = seq.strip().split()[:max_len]
    ids = [vocab.get(t, vocab[UNK]) for t in toks]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    return ids


MAX_LEN = 40


# ---------- PyTorch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode_sequence(row["sequence"], MAX_LEN), dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"input_ids": ids, "labels": label, "sequence": row["sequence"]}


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["sequence"] for b in batch]
    return {"input_ids": input_ids, "labels": labels, "sequence": seqs}


train_loader = DataLoader(
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
        emb = self.embed(x)  # B,L,D
        mask = (x != pad_idx).unsqueeze(-1).float()
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-6)
        pooled = summed / denom
        return self.fc(pooled)


num_labels = len(set(spr["train"]["label"]))
model = MeanEmbedClassifier(len(vocab), 128, num_labels).to(device)

# ---------- optimizer & loss ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


# ---------- augmentation (shape renaming) ----------
def shape_rename(seq):
    toks = seq.strip().split()
    shapes = list({t[0] for t in toks})
    mapping = {s: random.choice(string.ascii_uppercase) for s in shapes}
    new_toks = [mapping[t[0]] + t[1:] if len(t) > 1 else mapping[t[0]] for t in toks]
    return " ".join(new_toks)


# ---------- metrics storage ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_ACS": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- training loop ----------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        # optional on-the-fly augmentation 50% probability
        seqs_aug = []
        for s in batch["sequence"]:
            if random.random() < 0.5:
                s = shape_rename(s)
            seqs_aug.append(s)
        input_ids = torch.stack(
            [
                torch.tensor(encode_sequence(s, MAX_LEN), dtype=torch.long)
                for s in seqs_aug
            ]
        )
        batch["input_ids"] = input_ids

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append((epoch, train_loss))

    # ----- validation -----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dev_loader:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["input_ids"])
            loss = criterion(logits, batch_t["labels"])
            val_loss += loss.item()
            preds = logits.argmax(-1).cpu().tolist()
            experiment_data["SPR_BENCH"]["predictions"].extend(preds)
            experiment_data["SPR_BENCH"]["ground_truth"].extend(
                batch["labels"].cpu().tolist()
            )
    val_loss /= len(dev_loader)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append((epoch, val_loss))

    # ----- ACS computation (subset for speed) -----
    def compute_ACS(model, dataset, max_samples=1000, n_aug=5):
        model.eval()
        total = 0
        consist_sum = 0.0
        with torch.no_grad():
            for i in range(min(len(dataset), max_samples)):
                row = dataset[i]
                base_seq = row["sequence"]
                label = row["label"]
                seq_variants = [base_seq] + [
                    shape_rename(base_seq) for _ in range(n_aug)
                ]
                correct = 0
                for sv in seq_variants:
                    ids = (
                        torch.tensor(encode_sequence(sv, MAX_LEN), dtype=torch.long)
                        .unsqueeze(0)
                        .to(device)
                    )
                    pred = model(ids).argmax(-1).item()
                    if pred == label:
                        correct += 1
                consist_sum += correct / len(seq_variants)
                total += 1
        return consist_sum / total if total > 0 else 0.0

    val_ACS = compute_ACS(model, spr["dev"])
    experiment_data["SPR_BENCH"]["metrics"]["val_ACS"].append((epoch, val_ACS))

    print(
        f"Epoch {epoch}: train_loss = {train_loss:.4f} | validation_loss = {val_loss:.4f} | val_ACS = {val_ACS:.4f}"
    )

# ---------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
