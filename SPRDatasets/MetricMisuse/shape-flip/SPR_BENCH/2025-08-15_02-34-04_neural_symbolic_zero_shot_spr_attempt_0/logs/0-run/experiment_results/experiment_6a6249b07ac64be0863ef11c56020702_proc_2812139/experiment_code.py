import os, pathlib, time, json, math, random, warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# -------------------- dataset location helper ---------------------
def find_spr_root() -> pathlib.Path:
    candidates = []
    env_path = os.getenv("SPR_DIR")
    if env_path:
        candidates.append(pathlib.Path(env_path))
    candidates.append(pathlib.Path.cwd() / "SPR_BENCH")
    for parent in pathlib.Path.cwd().resolve().parents:
        candidates.append(parent / "SPR_BENCH")
    for cand in candidates:
        if (cand / "train.csv").exists():
            print(f"Found SPR_BENCH at: {cand}")
            return cand
    raise FileNotFoundError(
        "Unable to locate SPR_BENCH dataset. "
        "Set $SPR_DIR or place SPR_BENCH folder in the current/parent directory."
    )


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


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------------- Dataset class --------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, token2idx, label2idx, max_len=30):
        self.data = hf_split
        self.tok2id = token2idx
        self.lab2id = label2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        ids = [
            self.tok2id.get(tok, self.tok2id["<unk>"]) for tok in seq.strip().split()
        ]
        ids = ids[: self.max_len]
        pad_len = self.max_len - len(ids)
        return ids + [self.tok2id["<pad>"]] * pad_len, len(ids)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids, real_len = self.encode(row["sequence"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(real_len, dtype=torch.long),
            "label": torch.tensor(self.lab2id[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


# -------------------- model --------------------
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


# -------------------- prepare data --------------------
spr_root = find_spr_root()
spr = load_spr_bench(spr_root)
specials = ["<pad>", "<unk>"]
vocab_set = set()
for s in spr["train"]["sequence"]:
    vocab_set.update(s.strip().split())
token2idx = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab_set))}
for i, tok in enumerate(specials):
    token2idx[tok] = i
pad_idx = token2idx["<pad>"]
labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

train_ds = SPRDataset(spr["train"], token2idx, label2idx)
dev_ds = SPRDataset(spr["dev"], token2idx, label2idx)
test_ds = SPRDataset(spr["test"], token2idx, label2idx)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# -------------------- training utils --------------------
def run_epoch(loader, model, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            total += batch["label"].size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())
            all_seqs.extend(batch["raw_seq"])
    avg_loss = total_loss / total
    y_true = [idx2label[i] for i in all_labels]
    y_pred = [idx2label[i] for i in all_preds]
    swa = shape_weighted_accuracy(all_seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(all_seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0
    return avg_loss, (swa, cwa, hwa), y_true, y_pred


# -------------------- hyperparameter sweep --------------------
learning_rates = [3e-4, 1e-3, 3e-3]
num_epochs = 5
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"learning_rate": {}}

for lr in learning_rates:
    print(f"\n===== Training with learning_rate={lr} =====")
    lr_key = f"{lr:.0e}" if lr < 1 else str(lr)
    experiment_data["learning_rate"][lr_key] = {
        "spr_bench": {
            "metrics": {"train": [], "val": [], "test": None},
            "losses": {"train": [], "val": [], "test": None},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
    # New model/optimizer
    model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_met, _, _ = run_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_met, _, _ = run_epoch(dev_loader, model, criterion, None)

        ed = experiment_data["learning_rate"][lr_key]["spr_bench"]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(tr_met)
        ed["metrics"]["val"].append(val_met)
        ed["timestamps"].append(time.time())

        print(
            f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  "
            f"SWA={val_met[0]:.4f} CWA={val_met[1]:.4f} HWA={val_met[2]:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    # --- Test evaluation ---
    test_loss, test_met, y_true_test, y_pred_test = run_epoch(
        test_loader, model, criterion, None
    )
    print(
        f"Test -> SWA={test_met[0]:.4f}  CWA={test_met[1]:.4f}  HWA={test_met[2]:.4f}"
    )

    ed = experiment_data["learning_rate"][lr_key]["spr_bench"]
    ed["losses"]["test"] = test_loss
    ed["metrics"]["test"] = test_met
    ed["predictions"] = y_pred_test
    ed["ground_truth"] = y_true_test

    # Plot losses
    fig, ax = plt.subplots()
    ax.plot(ed["losses"]["train"], label="train")
    ax.plot(ed["losses"]["val"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"SPR GRU Loss (lr={lr})")
    ax.legend()
    plt.savefig(os.path.join(working_dir, f"spr_loss_curve_lr{lr_key}.png"))
    plt.close(fig)

# -------------------- save experiment data --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nAll outputs saved to {working_dir}")
