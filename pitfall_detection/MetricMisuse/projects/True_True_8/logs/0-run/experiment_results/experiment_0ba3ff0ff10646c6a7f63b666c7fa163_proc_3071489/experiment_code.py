import os, random, string, pathlib, time
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# -------------------------------------------------------
# basic setup & reproducibility
# -------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# -------------------------------------------------------
# data loading helpers
# -------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_synthetic_dataset(n_train=2000, n_dev=500, n_test=500, max_len=12):
    def _gen_row():
        l = random.randint(4, max_len)
        seq, label = [], 0
        for _ in range(l):
            sh = random.choice(string.ascii_uppercase[:5])
            co = random.choice("01234")
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
print("Loaded dataset sizes ->", {k: len(v) for k, v in spr.items()})

# -------------------------------------------------------
# vocabulary
# -------------------------------------------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]
MAX_LEN = 40


def encode_sequence(seq: str, max_len: int = MAX_LEN):
    ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# -------------------------------------------------------
# metrics
# -------------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1, sum(weights))


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1, sum(weights))


def composite_weighted_accuracy(sequences, y_true, y_pred):
    return 0.5 * (
        shape_weighted_accuracy(sequences, y_true, y_pred)
        + color_weighted_accuracy(sequences, y_true, y_pred)
    )


# -------------------------------------------------------
# torch dataset wrapper
# -------------------------------------------------------
class SPRTorchDataset(TorchDataset):
    def __init__(self, hf_ds: HFDataset):
        self.hf_ds = hf_ds

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        row = self.hf_ds[idx]
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


def shape_rename(seq: str):
    toks = seq.split()
    shapes = list({t[0] for t in toks})
    mapping = {s: random.choice(string.ascii_uppercase) for s in shapes}
    return " ".join([mapping[t[0]] + t[1:] for t in toks])


# -------------------------------------------------------
# model
# -------------------------------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz: int, embed_dim: int, num_cls: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_cls)

    def forward(self, x):
        emb = self.embed(x)  # B,L,D
        mask = (x != pad_idx).unsqueeze(-1).float()
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc(pooled)


# -------------------------------------------------------
# experiment store
# -------------------------------------------------------
experiment_data = {"embed_dim_sweep": {}}

# -------------------------------------------------------
# training loop
# -------------------------------------------------------
embed_dims = [64, 128, 256]
EPOCHS = 5
BATCH_SZ = 128

for dim in embed_dims:
    print(f"\n===== Training embed_dim={dim} =====")
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]),
        batch_size=BATCH_SZ,
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    run_store = {
        "losses": {"train": [], "val": []},
        "metrics": {"CoWA_train": [], "CoWA_val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---------- training ----------
        model.train()
        t0 = time.time()
        train_loss_total = 0.0
        for batch in train_loader:
            # on-the-fly augmentation (50% chance per sequence)
            aug_sequences = [
                shape_rename(s) if random.random() < 0.5 else s
                for s in batch["sequence"]
            ]
            batch["input_ids"] = torch.stack(
                [
                    torch.tensor(encode_sequence(s), dtype=torch.long)
                    for s in aug_sequences
                ]
            )
            # move tensors to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader)

        # ---------- evaluation ----------
        model.eval()
        val_loss_total = 0.0
        val_sequences, val_true, val_pred = [], [], []
        train_sequences, train_true, train_pred = [], [], []

        # compute train metrics on one random mini-batch (cheap)
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            sample_batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample_batch.items()
            }
            logits = model(sample_batch_t["input_ids"])
            preds = logits.argmax(-1).cpu().tolist()
            train_sequences.extend(sample_batch["sequence"])
            train_true.extend(sample_batch["labels"].tolist())
            train_pred.extend(preds)

        # full dev set
        with torch.no_grad():
            for batch in dev_loader:
                batch_t = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch_t["input_ids"])
                val_loss_total += criterion(logits, batch_t["labels"]).item()
                preds = logits.argmax(-1).cpu().tolist()

                val_sequences.extend(batch["sequence"])
                val_true.extend(batch["labels"].tolist())
                val_pred.extend(preds)

        val_loss = val_loss_total / len(dev_loader)

        # metrics
        CoWA_train = composite_weighted_accuracy(
            train_sequences, train_true, train_pred
        )
        CoWA_val = composite_weighted_accuracy(val_sequences, val_true, val_pred)

        run_store["losses"]["train"].append((epoch, train_loss))
        run_store["losses"]["val"].append((epoch, val_loss))
        run_store["metrics"]["CoWA_train"].append((epoch, CoWA_train))
        run_store["metrics"]["CoWA_val"].append((epoch, CoWA_val))
        run_store["predictions"].append((epoch, val_pred))
        run_store["ground_truth"].append((epoch, val_true))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"CoWA_train={CoWA_train:.4f}  CoWA_val={CoWA_val:.4f} | "
            f"time={(time.time()-t0):.1f}s"
        )

    experiment_data["embed_dim_sweep"][f"emb_{dim}"] = run_store

# -------------------------------------------------------
# save experiment data
# -------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
