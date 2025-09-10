# -------------------- imports & misc --------------------
import os, pathlib, time, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- dataset helpers --------------------
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
        "SPR_BENCH dataset not found. Set $SPR_DIR or place folder appropriately."
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


def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------------- Dataset --------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, tok2id, lab2id, max_len=30):
        self.data = hf_split
        self.tok2id = tok2id
        self.lab2id = lab2id
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
            "input_ids": torch.tensor(ids),
            "lengths": torch.tensor(real_len),
            "label": torch.tensor(self.lab2id[row["label"]]),
            "raw_seq": row["sequence"],
        }


# -------------------- Model --------------------
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


# -------------------- data prep --------------------
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

train_loader_fn = lambda bs: DataLoader(train_ds, batch_size=bs, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# -------------------- run helpers --------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(bt["input_ids"], bt["lengths"])
            loss = criterion(logits, bt["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * bt["label"].size(0)
            total += bt["label"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(bt["label"].cpu().numpy())
            all_seqs.extend(bt["raw_seq"])
    avg_loss = total_loss / total
    y_true = [idx2label[i] for i in all_labels]
    y_pred = [idx2label[i] for i in all_preds]
    swa = shape_weighted_accuracy(all_seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(all_seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    return avg_loss, (swa, cwa, hwa), y_true, y_pred


# -------------------- hyperparameter sweep --------------------
lr_grid = [3e-4, 5e-4, 1e-3, 2e-3]
num_epochs = 5
batch_size = 256

experiment_data = {"learning_rate": {}}
best_dev_hwa = -1
best_run_key = None
best_state = None

for lr in lr_grid:
    print(f"\n--- Training with learning_rate={lr} ---")
    model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tr_losses = []
    val_losses = []
    tr_metrics = []
    val_metrics = []
    for ep in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_met, _, _ = run_epoch(
            model, train_loader_fn(batch_size), criterion, optimizer
        )
        val_loss, val_met, _, _ = run_epoch(model, dev_loader, criterion)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        tr_metrics.append(tr_met)
        val_metrics.append(val_met)
        print(
            f"Epoch {ep} | val_loss {val_loss:.4f} HWA {val_met[2]:.4f} ({time.time()-t0:.1f}s)"
        )
    # store
    run_key = f"lr_{lr}"
    experiment_data["learning_rate"][run_key] = {
        "metrics": {"train": tr_metrics, "val": val_metrics},
        "losses": {"train": tr_losses, "val": val_losses},
    }
    # track best
    if val_metrics[-1][2] > best_dev_hwa:
        best_dev_hwa = val_metrics[-1][2]
        best_run_key = run_key
        best_state = model.state_dict()

# -------------------- evaluate best on test --------------------
print(f"\nBest learning rate run: {best_run_key} (dev HWA={best_dev_hwa:.4f})")
# rebuild model with same lr to load state
best_lr = float(best_run_key.split("_")[1])
best_model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
best_model.load_state_dict(best_state)
criterion = nn.CrossEntropyLoss()
test_loss, test_met, y_true_test, y_pred_test = run_epoch(
    best_model, test_loader, criterion
)
print(f"Test -> SWA={test_met[0]:.4f}  CWA={test_met[1]:.4f}  HWA={test_met[2]:.4f}")

experiment_data["learning_rate"][best_run_key]["losses"]["test"] = test_loss
experiment_data["learning_rate"][best_run_key]["metrics"]["test"] = test_met
experiment_data["learning_rate"][best_run_key]["predictions"] = y_pred_test
experiment_data["learning_rate"][best_run_key]["ground_truth"] = y_true_test

# -------------------- save arrays --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# optional plots for best run
fig, ax = plt.subplots()
ax.plot(
    experiment_data["learning_rate"][best_run_key]["losses"]["train"], label="train"
)
ax.plot(experiment_data["learning_rate"][best_run_key]["losses"]["val"], label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"Loss (best lr={best_lr})")
ax.legend()
plt.savefig(os.path.join(working_dir, "loss_curve_best_lr.png"))
plt.close(fig)

print(f"All outputs saved to {working_dir}")
