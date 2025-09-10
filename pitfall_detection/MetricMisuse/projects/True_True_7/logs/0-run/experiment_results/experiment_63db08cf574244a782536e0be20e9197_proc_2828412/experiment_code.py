import os, pathlib, time, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------- storage dict --------------------
experiment_data = {
    "random_token_mask_15": {"SPR_BENCH": {"runs": {}}}  # to be filled with epochs_k
}

# -------------------- I/O & misc --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- dataset location helper --------------------
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
        "Set $SPR_DIR or place SPR_BENCH in cwd/parent."
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


# -------------------- metrics helpers --------------------
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
unk_idx = token2idx["<unk>"]

labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

train_ds = SPRDataset(spr["train"], token2idx, label2idx)
dev_ds = SPRDataset(spr["dev"], token2idx, label2idx)
test_ds = SPRDataset(spr["test"], token2idx, label2idx)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# -------------------- training / eval loop with masking --------------------
def random_token_mask(x, pad_idx, unk_idx, prob=0.15):
    maskable = x != pad_idx
    rand = torch.rand_like(x.float())
    mask = (rand < prob) & maskable
    x_masked = x.clone()
    x_masked[mask] = unk_idx
    return x_masked


def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.set_grad_enabled(train_flag):
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            inp_ids = batch["input_ids"]
            if train_flag:  # apply 15% token masking
                inp_ids = random_token_mask(inp_ids, pad_idx, unk_idx, 0.15)
            logits = model(inp_ids, batch["lengths"])
            loss = criterion(logits, batch["label"])
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())
            all_seqs.extend(batch["raw_seq"])
    avg_loss = total_loss / total
    y_true = [idx2label[i] for i in all_labels]
    y_pred = [idx2label[i] for i in all_preds]
    swa = shape_weighted_accuracy(all_seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(all_seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    return avg_loss, (swa, cwa, hwa), y_true, y_pred


# -------------------- hyperparameter tuning --------------------
epoch_options = [5, 10, 20, 30]
patience = 3

for num_epochs in epoch_options:
    print(f"\n=== Training with num_epochs={num_epochs} (RandomMask15) ===")
    torch.cuda.empty_cache()
    model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_data = {
        "losses": {"train": [], "val": [], "test": None},
        "metrics": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    best_val_hwa, epochs_no_improve = -1.0, 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_met, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_met, _, _ = run_epoch(model, dev_loader, criterion)

        run_data["losses"]["train"].append(tr_loss)
        run_data["losses"]["val"].append(val_loss)
        run_data["metrics"]["train"].append(tr_met)
        run_data["metrics"]["val"].append(val_met)
        run_data["timestamps"].append(time.time())

        if val_met[2] > best_val_hwa:
            best_val_hwa = val_met[2]
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch}/{num_epochs}  val_loss={val_loss:.4f} "
            f"SWA={val_met[0]:.4f} CWA={val_met[1]:.4f} HWA={val_met[2]:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # reload best params before test
    model.load_state_dict(best_state)
    test_loss, test_met, y_true_test, y_pred_test = run_epoch(
        model, test_loader, criterion
    )
    run_data["losses"]["test"] = test_loss
    run_data["metrics"]["test"] = test_met
    run_data["predictions"] = y_pred_test
    run_data["ground_truth"] = y_true_test

    run_key = f"epochs_{num_epochs}"
    experiment_data["random_token_mask_15"]["SPR_BENCH"]["runs"][run_key] = run_data
    print(f"Test HWA={test_met[2]:.4f}")

# -------------------- save experiment data --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nAll results saved to {working_dir}/experiment_data.npy")

# (Optional) plot val loss curves for each setting
fig, ax = plt.subplots()
for k, v in experiment_data["random_token_mask_15"]["SPR_BENCH"]["runs"].items():
    ax.plot(v["losses"]["val"], label=k)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Loss")
ax.set_title("SPR GRU Loss with 15% Random Token Masking")
ax.legend()
plt.savefig(os.path.join(working_dir, "spr_loss_curves.png"))
plt.close(fig)
print("Loss plots saved.")
