import os, pathlib, time, math, json

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ----------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- robust data finder -----------------
def locate_spr_bench() -> pathlib.Path:
    """Return a valid SPR_BENCH directory or raise FileNotFoundError."""
    # 1) explicit env var
    cand = os.getenv("SPR_DIR")
    if cand and pathlib.Path(cand).is_dir():
        return pathlib.Path(cand)
    # 2) ./SPR_BENCH
    cand = pathlib.Path.cwd() / "SPR_BENCH"
    if cand.is_dir():
        return cand
    # 3) walk up max 4 levels
    base = pathlib.Path.cwd()
    for _ in range(4):
        base = base.parent
        cand = base / "SPR_BENCH"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Unable to locate SPR_BENCH folder. Set SPR_DIR env variable or place "
        "the folder in the current (or parent) directory."
    )


# ----------------- dataset loading -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_shape_variety(s) for s in sequences]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_color_variety(s) for s in sequences]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ----------------- Dataset class -----------------
class SPRDataset(Dataset):
    def __init__(self, split, token2idx, label2idx, max_len=30):
        self.data = split
        self.t2i = token2idx
        self.l2i = label2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        ids = [self.t2i.get(tok, self.t2i["<unk>"]) for tok in seq.strip().split()]
        ids = ids[: self.max_len]
        pad = [self.t2i["<pad>"]] * (self.max_len - len(ids))
        return ids + pad, len(ids)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids, real_len = self.encode(row["sequence"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(real_len, dtype=torch.long),
            "label": torch.tensor(self.l2i[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


# ----------------- Model -----------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_cls, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, n_cls)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last)


# ----------------- Load data -----------------
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# vocab and label maps
specials = ["<pad>", "<unk>"]
vocab = {tok for seq in spr["train"]["sequence"] for tok in seq.split()}
token2idx = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab))}
for i, tok in enumerate(specials):
    token2idx[tok] = i
pad_idx = token2idx["<pad>"]
labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

# datasets / loaders
train_ds = SPRDataset(spr["train"], token2idx, label2idx)
dev_ds = SPRDataset(spr["dev"], token2idx, label2idx)
test_ds = SPRDataset(spr["test"], token2idx, label2idx)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)

# ----------------- Model / Optim / Loss -----------------
model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------- experiment data store -----------------
experiment_data = {
    "spr_bench": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ----------------- training helpers -----------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, n_items = 0.0, 0
    preds_all, labels_all, seqs_all = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            # move tensors
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["input_ids"], batch["lengths"])
            loss = criterion(out, batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            n_items += batch["label"].size(0)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(batch["label"].cpu().numpy())
            seqs_all.extend(batch["raw_seq"])
    avg_loss = total_loss / n_items
    y_true = [idx2label[i] for i in labels_all]
    y_pred = [idx2label[i] for i in preds_all]
    swa = shape_weighted_accuracy(seqs_all, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs_all, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0
    return avg_loss, (swa, cwa, hwa), y_true, y_pred


# ----------------- training loop -----------------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    start = time.time()
    tr_loss, tr_metric, _, _ = run_epoch(train_loader, True)
    val_loss, val_metric, _, _ = run_epoch(dev_loader, False)
    experiment_data["spr_bench"]["losses"]["train"].append(tr_loss)
    experiment_data["spr_bench"]["losses"]["val"].append(val_loss)
    experiment_data["spr_bench"]["metrics"]["train"].append(tr_metric)
    experiment_data["spr_bench"]["metrics"]["val"].append(val_metric)
    experiment_data["spr_bench"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}, "
        f"SWA={val_metric[0]:.4f}, CWA={val_metric[1]:.4f}, HWA={val_metric[2]:.4f} "
        f"(elapsed {time.time()-start:.1f}s)"
    )

# ----------------- final test -----------------
test_loss, test_metric, y_true_test, y_pred_test = run_epoch(test_loader, False)
print(
    f"\nTest: SWA={test_metric[0]:.4f} CWA={test_metric[1]:.4f} HWA={test_metric[2]:.4f}"
)
experiment_data["spr_bench"]["losses"]["test"] = test_loss
experiment_data["spr_bench"]["metrics"]["test"] = test_metric
experiment_data["spr_bench"]["predictions"] = y_pred_test
experiment_data["spr_bench"]["ground_truth"] = y_true_test

# ----------------- save -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ----------------- plot -----------------
fig, ax = plt.subplots()
ax.plot(experiment_data["spr_bench"]["losses"]["train"], label="train")
ax.plot(experiment_data["spr_bench"]["losses"]["val"], label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("SPR GRU loss curve")
ax.legend()
plt.savefig(os.path.join(working_dir, "spr_loss_curve.png"))
plt.close(fig)
print(f"Outputs saved to {working_dir}")
