import os, pathlib, random, time, json, math

# ---------------- working dir -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- std / torch imports ----------------------------------------
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

# ---------------- device ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- SPR helpers -------------------------------------------------
try:
    from SPR import (
        load_spr_bench,
        shape_weighted_accuracy,
        color_weighted_accuracy,
    )
except ImportError:
    # Minimal re-implementation so the file is standalone
    from datasets import load_dataset, DatasetDict

    def load_spr_bench(root: pathlib.Path):
        def _load(csv_name):
            return load_dataset(
                "csv",
                data_files=str(root / csv_name),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = DatasetDict()
        for sp in ["train", "dev", "test"]:
            d[sp] = _load(f"{sp}.csv")
        return d

    def _count_variety(sequence, idx):
        return len(
            set(token[idx] for token in sequence.strip().split() if len(token) > idx)
        )

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_variety(s, 0) for s in seqs]
        correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
        return sum(correct) / sum(w) if sum(w) else 0.0

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_variety(s, 1) for s in seqs]
        correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
        return sum(correct) / sum(w) if sum(w) else 0.0


# ---------------- load / make dataset ----------------------------------------
DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "SPR_BENCH"))
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH not found, generating synthetic toy data.")
    # Use HF dataset object but keep alias to avoid name clash
    from datasets import Dataset as HFDataset, DatasetDict

    TOKEN_CHOICES = ["SC", "SR", "TC", "TR", "HC", "HR"]
    LABEL_CHOICES = ["A", "B"]

    def make_split(n_rows):
        sequences = [
            " ".join(random.choice(TOKEN_CHOICES) for _ in range(random.randint(3, 7)))
            for _ in range(n_rows)
        ]
        return {
            "id": [str(i) for i in range(n_rows)],
            "sequence": sequences,
            "label": [random.choice(LABEL_CHOICES) for _ in range(n_rows)],
        }

    spr = DatasetDict()
    for split, n in [("train", 200), ("dev", 60), ("test", 60)]:
        spr[split] = HFDataset.from_dict(make_split(n))


# ---------------- vocab / label maps -----------------------------------------
def tokenize(seq: str):
    return seq.strip().split()


all_tokens = set()
for seq in spr["train"]["sequence"]:
    all_tokens.update(tokenize(seq))
token2id = {tok: i for i, tok in enumerate(sorted(all_tokens))}
vocab_size = len(token2id)
print(f"Vocabulary size: {vocab_size}")

labels = sorted(set(spr["train"]["label"]))
label2id = {lab: i for i, lab in enumerate(labels)}
num_labels = len(labels)
print(f"Number of labels: {num_labels}")


# ---------------- PyTorch dataset wrapper ------------------------------------
class SPRVectorDataset(TorchDataset):
    def __init__(self, hf_dataset, token2id, label2id):
        self.seqs = hf_dataset["sequence"]
        self.X = [self._vectorize(seq, token2id) for seq in self.seqs]
        self.y = [label2id[lbl] for lbl in hf_dataset["label"]]

    @staticmethod
    def _vectorize(sequence: str, token2id: dict):
        vec = np.zeros(len(token2id), dtype=np.float32)
        for tok in set(tokenize(sequence)):  # multi-hot
            if tok in token2id:
                vec[token2id[tok]] = 1.0
        return vec

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "input": torch.tensor(self.X[idx], dtype=torch.float32),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
            "sequence": self.seqs[idx],
        }


train_set = SPRVectorDataset(spr["train"], token2id, label2id)
dev_set = SPRVectorDataset(spr["dev"], token2id, label2id)
test_set = SPRVectorDataset(spr["test"], token2id, label2id)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=256, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)


# ---------------- simple MLP model -------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(vocab_size, 256, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------- experiment data logging ------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------------- training loop ----------------------------------------------
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    # ---- training -----------------------------------------------------------
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- validation ---------------------------------------------------------
    model.eval()
    val_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            trues.extend(batch["label"].cpu().tolist())
    val_loss /= len(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    # ---- metrics ------------------------------------------------------------
    pred_labels = [labels[i] for i in preds]
    true_labels = [labels[i] for i in trues]
    swa = shape_weighted_accuracy(dev_set.seqs, true_labels, pred_labels)
    cwa = color_weighted_accuracy(dev_set.seqs, true_labels, pred_labels)
    hwa = 0.0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"swa": swa, "cwa": cwa, "hwa": hwa}
    )
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  "
        f"validation_loss={val_loss:.4f}  HWA={hwa:.4f}"
    )

# ---------------- test evaluation --------------------------------------------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input"])
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        trues.extend(batch["label"].cpu().tolist())

pred_labels = [labels[i] for i in preds]
true_labels = [labels[i] for i in trues]
swa = shape_weighted_accuracy(test_set.seqs, true_labels, pred_labels)
cwa = color_weighted_accuracy(test_set.seqs, true_labels, pred_labels)
hwa = 0.0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)

print(f"\nTEST | SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["test"] = {"swa": swa, "cwa": cwa, "hwa": hwa}
experiment_data["SPR_BENCH"]["predictions"] = pred_labels
experiment_data["SPR_BENCH"]["ground_truth"] = true_labels

# ---------------- save everything -------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
