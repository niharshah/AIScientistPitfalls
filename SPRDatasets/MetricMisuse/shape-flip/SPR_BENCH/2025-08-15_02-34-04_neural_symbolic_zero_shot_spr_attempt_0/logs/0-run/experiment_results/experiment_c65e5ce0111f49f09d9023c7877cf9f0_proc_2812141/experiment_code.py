import os, pathlib, time, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ----------------------- misc -----------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------- dataset helpers ------------------
def find_spr_root() -> pathlib.Path:
    env_path = os.getenv("SPR_DIR")
    candidates = []
    if env_path:
        candidates.append(pathlib.Path(env_path))
    candidates.append(pathlib.Path.cwd() / "SPR_BENCH")
    for parent in pathlib.Path.cwd().resolve().parents:
        candidates.append(parent / "SPR_BENCH")
    for cand in candidates:
        if (cand / "train.csv").exists():
            print(f"Found SPR_BENCH at: {cand}")
            return cand
    raise FileNotFoundError("SPR_BENCH not found, set $SPR_DIR or place folder nearby.")


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


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ---------------- dataset class --------------------
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


# ---------------- model ----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_cls, pad_idx, dropout_p=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout_p)
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
        last = self.drop(last)
        return self.fc(last)


# ---------------- prepare data ---------------------
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

train_loader = lambda bs: DataLoader(train_ds, batch_size=bs, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# ---------------- training helpers -----------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, tot = 0.0, 0
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
            tot_loss += loss.item() * bt["label"].size(0)
            tot += bt["label"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(bt["label"].cpu().numpy())
            all_seqs.extend(bt["raw_seq"])
    avg_loss = tot_loss / tot
    y_true = [idx2label[i] for i in all_labels]
    y_pred = [idx2label[i] for i in all_preds]
    swa = shape_weighted_accuracy(all_seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(all_seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    return avg_loss, (swa, cwa, hwa), y_true, y_pred


# --------------- hyperparameter sweep --------------
drop_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
experiment_data = {"dropout_rate": {}}

for p in drop_rates:
    print(f"\n=== Training with dropout p={p} ===")
    model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx, dropout_p=p).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    records = {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "val_hwa": None,
        "test_hwa": None,
    }
    num_epochs = 5
    for ep in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_met, _, _ = run_epoch(
            model, train_loader(256), criterion, optimizer
        )
        val_loss, val_met, _, _ = run_epoch(model, dev_loader, criterion)
        records["losses"]["train"].append(tr_loss)
        records["losses"]["val"].append(val_loss)
        records["metrics"]["train"].append(tr_met)
        records["metrics"]["val"].append(val_met)
        print(
            f"Epoch {ep}  val_loss={val_loss:.4f}  SWA={val_met[0]:.4f}  CWA={val_met[1]:.4f}  HWA={val_met[2]:.4f} ({time.time()-t0:.1f}s)"
        )
    # test
    test_loss, test_met, y_true_test, y_pred_test = run_epoch(
        model, test_loader, criterion
    )
    print(f"Test  SWA={test_met[0]:.4f}  CWA={test_met[1]:.4f}  HWA={test_met[2]:.4f}")
    records["losses"]["test"] = test_loss
    records["metrics"]["test"] = test_met
    records["predictions"] = y_pred_test
    records["ground_truth"] = y_true_test
    records["val_hwa"] = records["metrics"]["val"][-1][2]
    records["test_hwa"] = test_met[2]
    experiment_data["dropout_rate"][str(p)] = records

# --------------- save everything -------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"\nAll results saved to {working_dir}/experiment_data.npy")
