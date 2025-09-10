import os, pathlib, json, math, time, sys
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------- I/O & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- robust SPR_BENCH finder --------------------
def find_spr_bench_dir() -> pathlib.Path:
    """
    Look for a folder that contains train.csv, dev.csv, test.csv.
    Search order:
    1. SPR_DIR env variable
    2. ./SPR_BENCH relative to cwd
    3. Any parent directory having SPR_BENCH
    """
    env_path = os.getenv("SPR_DIR")
    candidates = []
    if env_path:
        candidates.append(pathlib.Path(env_path))
    # direct sub-dir
    candidates.append(pathlib.Path.cwd() / "SPR_BENCH")
    # parents
    for parent in pathlib.Path.cwd().parents:
        candidates.append(parent / "SPR_BENCH")

    for cand in candidates:
        if cand.is_dir():
            needed = ["train.csv", "dev.csv", "test.csv"]
            if all((cand / f).is_file() for f in needed):
                print(f"Found SPR_BENCH at: {cand}")
                return cand.resolve()
    raise FileNotFoundError(
        "Could not locate SPR_BENCH directory. "
        "Set SPR_DIR env variable or place SPR_BENCH with csv files in cwd/parents."
    )


# -------------------- data loading utils --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# -------------------- dataset class --------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, token2idx, label2idx, max_len=30):
        self.data = hf_split
        self.token2idx = token2idx
        self.label2idx = label2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _encode_seq(self, seq: str):
        ids = [
            self.token2idx.get(tok, self.token2idx["<unk>"])
            for tok in seq.strip().split()
        ]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        pad = [self.token2idx["<pad>"]] * (self.max_len - len(ids))
        return ids + pad, len(ids)

    def __getitem__(self, idx):
        row = self.data[idx]
        token_ids, real_len = self._encode_seq(row["sequence"])
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "lengths": torch.tensor(real_len, dtype=torch.long),
            "label": torch.tensor(self.label2idx[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


# -------------------- model --------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # gather last timestep (for both directions)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last)


# -------------------- prepare data --------------------
DATA_PATH = find_spr_bench_dir()
spr = load_spr_bench(DATA_PATH)

# vocab
specials = ["<pad>", "<unk>"]
vocab = set()
for s in spr["train"]["sequence"]:
    vocab.update(s.strip().split())
token2idx = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab))}
for i, tok in enumerate(specials):
    token2idx[tok] = i
pad_idx = token2idx["<pad>"]

# labels
labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

train_ds = SPRDataset(spr["train"], token2idx, label2idx)
dev_ds = SPRDataset(spr["dev"], token2idx, label2idx)
test_ds = SPRDataset(spr["test"], token2idx, label2idx)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

# -------------------- model, loss, optimizer --------------------
model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------- experiment data store --------------------
experiment_data = {
    "spr_bench": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# -------------------- helpers --------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            # move tensors
            tensor_batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            raw_seq = batch["raw_seq"]

            logits = model(tensor_batch["input_ids"], tensor_batch["lengths"])
            loss = criterion(logits, tensor_batch["label"])

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = tensor_batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(tensor_batch["label"].cpu().numpy())
            all_seqs.extend(raw_seq)

    avg_loss = total_loss / total
    y_true = [idx2label[i] for i in all_labels]
    y_pred = [idx2label[i] for i in all_preds]
    swa = shape_weighted_accuracy(all_seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(all_seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0
    return avg_loss, (swa, cwa, hwa), y_true, y_pred


# -------------------- training loop --------------------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss, train_metrics, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_metrics, _, _ = run_epoch(dev_loader, train=False)

    experiment_data["spr_bench"]["losses"]["train"].append(train_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)
    experiment_data["spr_bench"]["metrics"]["train"].append(train_metrics)
    experiment_data["spr_bench"]["metrics"]["val"].append(val_metrics)
    experiment_data["spr_bench"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: "
        f"val_loss = {val_loss:.4f} "
        f"SWA = {val_metrics[0]:.4f} "
        f"CWA = {val_metrics[1]:.4f} "
        f"HWA = {val_metrics[2]:.4f}   "
        f"(elapsed {time.time() - t0:.1f}s)"
    )

# -------------------- test evaluation --------------------
test_loss, test_metrics, y_true_test, y_pred_test = run_epoch(test_loader, train=False)
print(
    "\nTest set -> "
    f"SWA={test_metrics[0]:.4f}  CWA={test_metrics[1]:.4f}  HWA={test_metrics[2]:.4f}"
)

experiment_data["spr_bench"]["losses"]["test"] = test_loss
experiment_data["spr_bench"]["metrics"]["test"] = test_metrics
experiment_data["spr_bench"]["predictions"] = y_pred_test
experiment_data["spr_bench"]["ground_truth"] = y_true_test

# -------------------- save experiment data --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# -------------------- visualize loss --------------------
fig, ax = plt.subplots()
ax.plot(experiment_data["spr_bench"]["losses"]["train"], label="train")
ax.plot(experiment_data["spr_bench"]["losses"]["val"], label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("SPR GRU Loss Curve")
ax.legend()
plt.savefig(os.path.join(working_dir, "spr_loss_curve.png"))
plt.close(fig)

print(f"All outputs saved in {working_dir}")
