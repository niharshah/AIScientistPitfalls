import os, random, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict, concatenate_datasets
import matplotlib.pyplot as plt
from typing import List, Dict

# -------------------------------------------------
# mandatory working dir and device setup
# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------
# experiment data container
# -------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "per_lr": {},  # lr -> {'metrics':…, 'losses':…}
        "best_lr": None,
        "test_macro_f1": None,
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------------------------------------
# deterministic seeding
# -------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()


# -------------------------------------------------
# dataset helpers
# -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=os.path.join(working_dir, ".cache_dsets"),
        )

    dset = DatasetDict()
    for sp in ["train", "dev", "test"]:
        dset[sp] = _load(f"{sp}.csv")
    return dset


def get_dataset() -> DatasetDict:
    # search a couple of likely locations
    candidate_roots = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for root in candidate_roots:
        if (root / "train.csv").exists():
            print(f"Loading real SPR_BENCH from {root}")
            return load_spr_bench(root)

    # ---------- synthetic fallback ----------
    print("SPR_BENCH not found – creating synthetic toy dataset")

    def _synth(n_rows):
        rows = []
        shapes = "ABCD"
        for i in range(n_rows):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            # toy rule: even # of 'A' → label 1 else 0
            rows.append(
                {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}
            )
        return rows

    def _to_ds(rows):
        # write rows to disk-less json dataset
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    dset = DatasetDict()
    dset["train"] = _to_ds(_synth(2000))
    dset["dev"] = _to_ds(_synth(500))
    dset["test"] = _to_ds(_synth(500))
    return dset


spr = get_dataset()

# -------------------------------------------------
# vocabulary and encoding utilities
# -------------------------------------------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(list(set(all_text)))
stoi = {ch: idx + 1 for idx, ch in enumerate(vocab)}  # 0 reserved for PAD/UNK
itos = {idx: ch for ch, idx in stoi.items()}
vocab_size = len(stoi) + 1
max_len = min(100, max(len(s) for s in spr["train"]["sequence"]))


def encode(seq: str) -> List[int]:
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.float),
        }


# -------------------------------------------------
# model definition
# -------------------------------------------------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int = 64, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.rnn(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


# -------------------------------------------------
# helpers for training / evaluation
# -------------------------------------------------
def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    epoch_loss, preds, labels = [], [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            # move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss.append(loss.item())
            preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
            labels.extend(batch["label"].long().cpu().numpy())
    macro_f1 = f1_score(labels, preds, average="macro")
    return float(np.mean(epoch_loss)), macro_f1


# -------------------------------------------------
# hyper-parameter sweep (learning-rate)
# -------------------------------------------------
batch_size = 128
lr_grid = [3e-4, 5e-4, 1e-3, 2e-3]
best_val_f1, best_lr, best_state = -1.0, None, None

for lr in lr_grid:
    print(f"\n--- Training with lr={lr} ---")
    model = CharBiGRU(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    tr_loader = DataLoader(
        SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size)

    metrics = {"train_macro_F1": [], "val_macro_F1": []}
    losses = {"train": [], "val": []}

    for epoch in range(1, 6):  # 5 epochs
        tr_loss, tr_f1 = run_epoch(model, tr_loader, criterion, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion)
        metrics["train_macro_F1"].append(tr_f1)
        metrics["val_macro_F1"].append(val_f1)
        losses["train"].append(tr_loss)
        losses["val"].append(val_loss)
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f} "
            f"macro_F1 = {val_f1:.4f}"
        )

    experiment_data["SPR_BENCH"]["per_lr"][lr] = {"metrics": metrics, "losses": losses}
    # track best
    if metrics["val_macro_F1"][-1] > best_val_f1:
        best_val_f1 = metrics["val_macro_F1"][-1]
        best_lr = lr
        best_state = model.state_dict()

experiment_data["SPR_BENCH"]["best_lr"] = best_lr
print(f"\nBest lr by val_macro_F1: {best_lr} ({best_val_f1:.4f})")

# -------------------------------------------------
# retrain on (train + dev) with best learning rate
# -------------------------------------------------
combined_train_ds = concatenate_datasets([spr["train"], spr["dev"]])
train_loader = DataLoader(
    SPRDataset(combined_train_ds), batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size)

best_model = CharBiGRU(vocab_size).to(device)
best_model.load_state_dict(best_state)  # warm-start
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)
criterion = nn.BCEWithLogitsLoss()

print("\nFine-tuning on combined train+dev (2 epochs)")
for epoch in range(1, 3):
    ft_loss, ft_f1 = run_epoch(best_model, train_loader, criterion, optimizer)
    print(
        f"Fine-tune Epoch {epoch}: train_loss = {ft_loss:.4f} "
        f"macro_F1 = {ft_f1:.4f}"
    )

# -------------------------------------------------
# final test evaluation
# -------------------------------------------------
best_model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = best_model(batch["input_ids"])
        test_preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
        test_labels.extend(batch["label"].long().cpu().numpy())

test_macro_f1 = f1_score(test_labels, test_preds, average="macro")
experiment_data["SPR_BENCH"]["test_macro_f1"] = test_macro_f1
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels
print(f"\nTest Macro-F1 with lr={best_lr}: {test_macro_f1:.4f}")

# -------------------------------------------------
# save artefacts
# -------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# optional: plot loss curve for best lr
import matplotlib

matplotlib.use("Agg")
plt.figure(figsize=(6, 4))
best_losses = experiment_data["SPR_BENCH"]["per_lr"][best_lr]["losses"]
plt.plot(best_losses["train"], label="train_loss")
plt.plot(best_losses["val"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title(f"Loss curve (lr={best_lr})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "best_lr_loss_curve.png"))
plt.close()
