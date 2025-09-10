import os, pathlib, time, random, math, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------------------------------------------- #
#  setup & utilities
# ------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ------------------------------------------------- #
#  locate SPR_BENCH
# ------------------------------------------------- #
def find_spr_root() -> pathlib.Path:
    if os.getenv("SPR_DIR"):
        cand = pathlib.Path(os.getenv("SPR_DIR"))
        if (cand / "train.csv").exists():
            return cand
    for p in [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents):
        cand = p / "SPR_BENCH"
        if (cand / "train.csv").exists():
            return cand
    raise FileNotFoundError(
        "Cannot locate SPR_BENCH; set $SPR_DIR env or place folder nearby."
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# ------------------------------------------------- #
#  metrics
# ------------------------------------------------- #
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ------------------------------------------------- #
#  dataset
# ------------------------------------------------- #
class SPRDataset(Dataset):
    def __init__(self, split, tok2id, lab2id, max_len=30):
        self.data = split
        self.tok2id = tok2id
        self.lab2id = lab2id
        self.max_len = max_len

    def encode_seq(self, seq):
        ids = [
            self.tok2id.get(tok, self.tok2id["<unk>"]) for tok in seq.strip().split()
        ]
        ids = ids[: self.max_len]
        padlen = self.max_len - len(ids)
        return ids + [self.tok2id["<pad>"]] * padlen, len(ids)

    # symbolic numeric features
    def sym_feats(self, seq):
        toks = seq.strip().split()
        shapes = [t[0] for t in toks if t]
        colors = [t[1] for t in toks if len(t) > 1]
        shape_var = len(set(shapes))
        color_var = len(set(colors))
        length = len(toks)
        ratio_shape = shape_var / (length + 1e-6)
        ratio_color = color_var / (length + 1e-6)
        shape_bigrams = len(set(a + b for a, b in zip(shapes, shapes[1:])))
        return [shape_var, color_var, length, ratio_shape, ratio_color, shape_bigrams]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids, length = self.encode_seq(row["sequence"])
        feats = self.sym_feats(row["sequence"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
            "sym_feats": torch.tensor(feats, dtype=torch.float),
            "label": torch.tensor(self.lab2id[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


# ------------------------------------------------- #
#  model
# ------------------------------------------------- #
class HybridClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, num_feats, n_cls, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hid_dim * 2 + num_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_cls),
        )

    def forward(self, ids, lengths, extra):
        emb = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(2))
        last = h.gather(1, idx).squeeze(1)
        concat = torch.cat([self.drop(last), extra], dim=1)
        return self.fc(concat)


# ------------------------------------------------- #
#  prepare data
# ------------------------------------------------- #
spr_root = find_spr_root()
spr = load_spr_bench(spr_root)

specials = ["<pad>", "<unk>"]
vocab = set()
for seq in spr["train"]["sequence"]:
    vocab.update(seq.strip().split())
token2id = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab))}
token2id["<pad>"] = 0
token2id["<unk>"] = 1
pad_idx = token2id["<pad>"]

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

train_ds = SPRDataset(spr["train"], token2id, lab2id)
dev_ds = SPRDataset(spr["dev"], token2id, lab2id)
test_ds = SPRDataset(spr["test"], token2id, lab2id)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# ------------------------------------------------- #
#  training helpers
# ------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    tot_loss, tot = 0.0, 0
    y_true, y_pred, seqs = [], [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"], batch["sym_feats"])
            loss = criterion(logits, batch["label"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = batch["label"].size(0)
            tot_loss += loss.item() * bs
            tot += bs
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["label"].cpu().numpy())
            seqs.extend(batch["raw_seq"])
    avg = tot_loss / tot
    y_true_lbl = [id2lab[i] for i in y_true]
    y_pred_lbl = [id2lab[i] for i in y_pred]
    swa = shape_weighted_accuracy(seqs, y_true_lbl, y_pred_lbl)
    return avg, swa, y_true_lbl, y_pred_lbl


# ------------------------------------------------- #
#  train
# ------------------------------------------------- #
num_epochs = 25
patience = 4
model = HybridClassifier(len(token2id), 64, 128, 6, len(labels), pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
    }
}

best_val_swa = -1.0
no_improve = 0
best_state = None

for epoch in range(1, num_epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_swa, _, _ = run_epoch(model, dev_loader, criterion)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)
    print(f"Epoch {epoch:02d}: validation_loss = {val_loss:.4f}  SWA = {val_swa:.4f}")
    # early stopping
    if val_swa > best_val_swa:
        best_val_swa = val_swa
        no_improve = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
    if no_improve >= patience:
        print("Early stopping triggered.")
        break

# reload best
model.load_state_dict(best_state)
test_loss, test_swa, y_true_test, y_pred_test = run_epoch(model, test_loader, criterion)
print(f"\nTEST  loss={test_loss:.4f}  SWA={test_swa:.4f}")
experiment_data["SPR_BENCH"]["losses"]["test"] = test_loss
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_swa
experiment_data["SPR_BENCH"]["predictions"] = y_pred_test
experiment_data["SPR_BENCH"]["ground_truth"] = y_true_test

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"All data saved to {working_dir}/experiment_data.npy")

# plot loss curve
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"])
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("Validation Loss")
plt.savefig(os.path.join(working_dir, "val_loss.png"))
plt.close()
