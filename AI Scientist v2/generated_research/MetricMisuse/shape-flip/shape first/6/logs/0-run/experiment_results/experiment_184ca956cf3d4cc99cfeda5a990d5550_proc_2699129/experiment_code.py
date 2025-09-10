import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from collections import Counter
from datasets import DatasetDict

# ------------------------------------------------------------------
# Required working directory & experiment dict
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "signatures": [],
    }
}
# ------------------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# ----------  SPR I/O (from given utility, slightly wrapped) --------
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


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ------------------------------------------------------------------
# -----------------  Fallback synthetic data  -----------------------
def build_synthetic_dataset(n_train=5000, n_dev=1000, n_test=1000) -> DatasetDict:
    shapes = "SCRTP"  # Square, Circle, Rectangle, Triangle, Pentagon
    colors = "RGBYM"  # Red, Green, Blue, Yellow, Magenta

    def rand_seq():
        length = random.randint(4, 12)
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )

    def gen_split(n):
        seqs, labels = [], []
        for _ in range(n):
            s = rand_seq()
            # arbitrary synthetic label rule: 1 if more unique shapes than colors else 0
            label = int(count_shape_variety(s) > count_color_variety(s))
            seqs.append(s)
            labels.append(label)
        return {"id": [str(i) for i in range(n)], "sequence": seqs, "label": labels}

    hf = DatasetDict()
    import datasets

    hf["train"] = datasets.Dataset.from_dict(gen_split(n_train))
    hf["dev"] = datasets.Dataset.from_dict(gen_split(n_dev))
    hf["test"] = datasets.Dataset.from_dict(gen_split(n_test))
    return hf


# ------------------------------------------------------------------
# ------------------  Load dataset (real or fake) -------------------
SPR_ROOT = pathlib.Path("./SPR_BENCH")
if SPR_ROOT.exists():
    print("Loading real SPR_BENCH dataset …")
    ds = load_spr_bench(SPR_ROOT)
else:
    print("Real SPR_BENCH not found, generating synthetic data …")
    ds = build_synthetic_dataset()

label_set = sorted(set(ds["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
num_labels = len(label2id)
print(f"Number of labels = {num_labels}")


# ------------------------------------------------------------------
# --------------------  Vocabulary ---------------------------------
def build_vocab(sequences: List[str]) -> Dict[str, int]:
    counter = Counter()
    for seq in sequences:
        counter.update(seq.strip().split())
    vocab = {
        tok: i + 2 for i, (tok, _) in enumerate(counter.most_common())
    }  # 0 PAD, 1 UNK
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


vocab = build_vocab(ds["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size = {vocab_size}")


def encode_seq(seq: str) -> List[int]:
    return [vocab.get(tok, 1) for tok in seq.strip().split()]


# ------------------------------------------------------------------
# -------------- PyTorch Dataset & DataLoader ----------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.ids = hf_split["id"]
        self.sequences = hf_split["sequence"]
        self.labels = [label2id[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "input": torch.tensor(encode_seq(self.sequences[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.sequences[idx],
        }


def collate_fn(batch):
    lengths = [len(item["input"]) for item in batch]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        padded[i, : lengths[i]] = item["input"]
    labels = torch.stack([b["label"] for b in batch])
    raw_seqs = [b["raw_seq"] for b in batch]
    ids = [b["id"] for b in batch]
    return {
        "ids": ids,
        "input": padded,
        "label": labels,
        "raw_seq": raw_seqs,
        "lengths": lengths,
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(ds["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    SPRTorchDataset(ds["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    SPRTorchDataset(ds["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# ------------------------------------------------------------------
# -------------------  Simple Mean-Pool model ----------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(self, x):  # x: (B, L)
        emb = self.emb(x)  # (B, L, D)
        mask = (x != 0).unsqueeze(-1)  # (B, L, 1)
        summed = (emb * mask).sum(1)  # (B, D)
        lengths = mask.sum(1).clamp(min=1)  # (B,1)
        mean = summed / lengths
        return self.classifier(mean)


model = MeanPoolClassifier(vocab_size, emb_dim=64, num_labels=num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------------------------------------
# ----------- Helper: rule signatures & NRGS -----------------------
def seq_signature(seq: str):
    return (len(seq.split()), count_shape_variety(seq), count_color_variety(seq))


train_signatures = set(seq_signature(s) for s in ds["train"]["sequence"])


def compute_metrics(all_raw, all_truth, all_pred):
    acc = np.mean(np.array(all_truth) == np.array(all_pred))
    swa = shape_weighted_accuracy(all_raw, all_truth, all_pred)
    cwa = color_weighted_accuracy(all_raw, all_truth, all_pred)
    # NRGS
    mask_unseen = [seq_signature(s) not in train_signatures for s in all_raw]
    if any(mask_unseen):
        true_unseen = [t for t, m in zip(all_truth, mask_unseen) if m]
        pred_unseen = [p for p, m in zip(all_pred, mask_unseen) if m]
        nrg = np.mean(np.array(true_unseen) == np.array(pred_unseen))
    else:
        nrg = float("nan")
    return {"acc": acc, "swa": swa, "cwa": cwa, "nrg": nrg}


# ------------------------------------------------------------------
# ------------------------- Training loop --------------------------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        inp = batch["input"].to(device)
        lbl = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inp.size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- validation ----
    model.eval()
    val_loss = 0.0
    all_raw, all_truth, all_pred = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input"].to(device)
            lbl = batch["label"].to(device)
            out = model(inp)
            loss = criterion(out, lbl)
            val_loss += loss.item() * inp.size(0)
            preds = out.argmax(1).cpu().tolist()
            truths = lbl.cpu().tolist()
            all_pred.extend(preds)
            all_truth.extend(truths)
            all_raw.extend(batch["raw_seq"])
    val_loss /= len(val_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    metrics = compute_metrics(all_raw, all_truth, all_pred)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(metrics)
    print(
        f'Epoch {epoch}: validation_loss = {val_loss:.4f} | ACC={metrics["acc"]:.3f} SWA={metrics["swa"]:.3f} CWA={metrics["cwa"]:.3f} NRGS={metrics["nrg"]:.3f}'
    )

# ------------------------------------------------------------------
# ------------------  Final evaluation on TEST ---------------------
model.eval()
all_raw, all_truth, all_pred = [], [], []
with torch.no_grad():
    for batch in test_loader:
        inp = batch["input"].to(device)
        lbl = batch["label"].to(device)
        out = model(inp)
        preds = out.argmax(1).cpu().tolist()
        truths = lbl.cpu().tolist()
        all_pred.extend(preds)
        all_truth.extend(truths)
        all_raw.extend(batch["raw_seq"])
test_metrics = compute_metrics(all_raw, all_truth, all_pred)
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_metrics
experiment_data["SPR_BENCH"]["predictions"] = all_pred
experiment_data["SPR_BENCH"]["ground_truth"] = all_truth
experiment_data["SPR_BENCH"]["signatures"] = [seq_signature(s) for s in all_raw]

print("\nTest set results:")
for k, v in test_metrics.items():
    print(f"  {k.upper():4s}: {v:.3f}")

# ------------------------------------------------------------------
# ------------------  Save experiment data -------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    f"\nAll experiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
)
