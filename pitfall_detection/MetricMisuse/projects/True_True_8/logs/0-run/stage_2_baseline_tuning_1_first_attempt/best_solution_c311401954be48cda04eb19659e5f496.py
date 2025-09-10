import os, random, string, pathlib, time, math, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict

# ---------- experiment dict ----------
experiment_data = {
    "batch_size": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},  # ACS etc.
            "losses": {"train": [], "val": []},  # CE losses
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper : load dataset (provided util) ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

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


# ---------- model ----------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)  # B,L,D
        mask = (x != pad_idx).unsqueeze(-1).float()
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc(pooled)


# ---------- augmentation ----------
def shape_rename(seq):
    toks = seq.strip().split()
    shapes = list({t[0] for t in toks})
    mapping = {s: random.choice(string.ascii_uppercase) for s in shapes}
    return " ".join(
        [mapping[t[0]] + t[1:] if len(t) > 1 else mapping[t[0]] for t in toks]
    )


# ---------- ACS ----------
def compute_ACS(model, dataset, max_samples=1000, n_aug=5):
    model.eval()
    total, consist_sum = 0, 0.0
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            base_seq = dataset[i]["sequence"]
            label = dataset[i]["label"]
            seq_variants = [base_seq] + [shape_rename(base_seq) for _ in range(n_aug)]
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
    return consist_sum / total if total else 0.0


# ---------- hyperparameter search ----------
batch_sizes = [32, 64, 128, 256, 512]
EPOCHS = 5
num_labels = len(set(spr["train"]["label"]))

for bs in batch_sizes:
    print(f"\n=== Training with batch_size={bs} ===")
    # loaders
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        SPRTorchDataset(spr["dev"]),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # model, opt
    model = MeanEmbedClassifier(len(vocab), 128, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for epoch in range(1, EPOCHS + 1):
        # ---- training ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # optional augmentation
            seqs_aug = [
                shape_rename(s) if random.random() < 0.5 else s
                for s in batch["sequence"]
            ]
            batch["input_ids"] = torch.stack(
                [
                    torch.tensor(encode_sequence(s, MAX_LEN), dtype=torch.long)
                    for s in seqs_aug
                ]
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
        experiment_data["batch_size"]["SPR_BENCH"]["losses"]["train"].append(
            (bs, epoch, train_loss)
        )

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch_t = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch_t["input_ids"])
                loss = criterion(logits, batch_t["labels"])
                val_loss += loss.item()
                preds = logits.argmax(-1).cpu().tolist()
                experiment_data["batch_size"]["SPR_BENCH"]["predictions"].extend(
                    [(bs, p) for p in preds]
                )
                experiment_data["batch_size"]["SPR_BENCH"]["ground_truth"].extend(
                    batch["labels"].cpu().tolist()
                )
        val_loss /= len(dev_loader)
        experiment_data["batch_size"]["SPR_BENCH"]["losses"]["val"].append(
            (bs, epoch, val_loss)
        )

        # ---- ACS ----
        val_ACS = compute_ACS(model, spr["dev"])
        experiment_data["batch_size"]["SPR_BENCH"]["metrics"]["val"].append(
            (bs, epoch, val_ACS)
        )

        print(
            f"[bs={bs}] Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ACS={val_ACS:.4f}"
        )

    # clean up GPU memory for next run
    del model, optimizer, train_loader, dev_loader
    torch.cuda.empty_cache()
    gc.collect()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
