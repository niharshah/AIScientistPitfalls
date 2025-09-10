import os, pathlib, time, json, math, random, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------- housekeeping --------------------
warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -------------------- dataset location helper --------------------
def find_spr_root() -> pathlib.Path:
    candidates, env_path = [], os.getenv("SPR_DIR")
    if env_path:
        candidates.append(pathlib.Path(env_path))
    candidates.append(pathlib.Path.cwd() / "SPR_BENCH")
    for p in pathlib.Path.cwd().resolve().parents:
        candidates.append(p / "SPR_BENCH")
    for cand in candidates:
        if (cand / "train.csv").exists():
            print(f"Found SPR_BENCH at: {cand}")
            return cand
    raise FileNotFoundError(
        "SPR_BENCH dataset not found (set $SPR_DIR or place folder)."
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
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


# -------------------- dataset class --------------------
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


# -------------------- prepare data common stuff --------------------
spr = load_spr_bench(find_spr_root())
specials = ["<pad>", "<unk>"]
vocab_set = set(tok for s in spr["train"]["sequence"] for tok in s.strip().split())
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


# -------------------- helper to run one training session --------------------
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
            bs = bt["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            preds = logits.argmax(dim=1).cpu().numpy()
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


def train_with_batch_size(batch_size, epochs=5):
    model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    losses_tr, losses_val, metrics_tr, metrics_val, times = [], [], [], [], []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        ltr, mtr, _, _ = run_epoch(model, tr_loader, criterion, optimizer)
        lval, mval, _, _ = run_epoch(model, dev_loader, criterion)
        losses_tr.append(ltr)
        losses_val.append(lval)
        metrics_tr.append(mtr)
        metrics_val.append(mval)
        times.append(time.time())
        print(
            f"[bs={batch_size}] Epoch {ep}/{epochs}  val_loss={lval:.4f}  HWA={mval[2]:.4f}  ({times[-1]-t0:.1f}s)"
        )
    # test
    ltest, mtest, y_true, y_pred = run_epoch(model, test_loader, criterion)
    return {
        "losses": {"train": losses_tr, "val": losses_val, "test": ltest},
        "metrics": {"train": metrics_tr, "val": metrics_val, "test": mtest},
        "predictions": y_pred,
        "ground_truth": y_true,
        "timestamps": times,
    }


# -------------------- hyper-parameter sweep --------------------
batch_sizes = [64, 128, 256, 512]
experiment_data = {"batch_size_tuning": {"spr_bench": {}}}

for bs in batch_sizes:
    print(f"\n==== Training with batch size {bs} ====")
    res = train_with_batch_size(bs, epochs=5)
    experiment_data["batch_size_tuning"]["spr_bench"][f"bs_{bs}"] = res
    # plot loss curve
    fig, ax = plt.subplots()
    ax.plot(res["losses"]["train"], label="train")
    ax.plot(res["losses"]["val"], label="val")
    ax.set_title(f"Loss (batch={bs})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(os.path.join(working_dir, f"loss_bs_{bs}.png"))
    plt.close(fig)

# -------------------- save experiment data --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nAll outputs stored in {working_dir}")
