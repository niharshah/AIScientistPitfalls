import os, pathlib, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ----------- WORK DIR & DEVICE -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------- DATASET FINDER -----------
def find_spr_root() -> pathlib.Path:
    env = os.getenv("SPR_DIR")
    cands = (
        ([pathlib.Path(env)] if env else [])
        + [pathlib.Path.cwd() / "SPR_BENCH"]
        + [p / "SPR_BENCH" for p in pathlib.Path.cwd().resolve().parents]
    )
    for c in cands:
        if (c / "train.csv").exists():
            print("Found SPR_BENCH at", c)
            return c
    raise FileNotFoundError("SPR_BENCH not found; set SPR_DIR or place folder nearby.")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


# ----------- METRICS -----------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ----------- DATASET CLASS -----------
class SPRDataset(Dataset):
    def __init__(self, split, tok2id, lab2id, max_len=30):
        self.data = split
        self.tok2id, self.lab2id, self.max_len = tok2id, lab2id, max_len

    def __len__(self):
        return len(self.data)

    def encode_tokens(self, seq):
        ids = [self.tok2id.get(tok, self.tok2id["<unk>"]) for tok in seq.split()]
        ids = ids[: self.max_len]
        return ids + [self.tok2id["<pad>"]] * (self.max_len - len(ids)), len(ids)

    def get_symbolic_feats(self, seq):
        n_shape = count_shape_variety(seq)
        n_color = len({tok[1] for tok in seq.split() if len(tok) > 1})
        length = len(seq.split())
        ratio = n_shape / (n_color + 1e-4)
        return [n_shape, n_color, length, ratio]

    def __getitem__(self, idx):
        row = self.data[idx]
        ids, lens = self.encode_tokens(row["sequence"])
        feats = self.get_symbolic_feats(row["sequence"])
        return {
            "input_ids": torch.tensor(ids),
            "lengths": torch.tensor(lens),
            "sym_feats": torch.tensor(feats, dtype=torch.float32),
            "label": torch.tensor(self.lab2id[row["label"]]),
            "raw_seq": row["sequence"],
        }


# ----------- MODEL -----------
class NeuroSymbolicSPR(nn.Module):
    def __init__(self, vocab, emb, hidden, n_cls, pad_idx, sym_dim=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.gru = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.sym_mlp = nn.Sequential(nn.Linear(sym_dim, hidden * 2), nn.ReLU())
        self.fc = nn.Linear(hidden * 4, n_cls)

    def forward(self, x, lens, sym):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (lens - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)  # [B, hidden*2]
        sym_vec = self.sym_mlp(sym)
        cat = torch.cat([last, sym_vec], dim=1)
        return self.fc(cat)


# ----------- PREPARE DATA -----------
root = find_spr_root()
spr = load_spr(root)

specials = ["<pad>", "<unk>"]
vocab = set()
for s in spr["train"]["sequence"]:
    vocab.update(s.split())
tok2id = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab))}
for i, tok in enumerate(specials):
    tok2id[tok] = i
pad_idx = tok2id["<pad>"]

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

train_ds = SPRDataset(spr["train"], tok2id, lab2id)
dev_ds = SPRDataset(spr["dev"], tok2id, lab2id)
test_ds = SPRDataset(spr["test"], tok2id, lab2id)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)


# ----------- TRAIN / EVAL LOOP -----------
def run_epoch(model, loader, crit, opt=None):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    total_loss, total = 0, 0
    all_pred, all_lab, all_seq = [], [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["input_ids"], batch["lengths"], batch["sym_feats"])
            loss = crit(out, batch["label"])
            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            all_pred.extend(out.argmax(1).cpu().numpy())
            all_lab.extend(batch["label"].cpu().numpy())
            all_seq.extend(batch["raw_seq"])
    avg_loss = total_loss / total
    y_true = [id2lab[i] for i in all_lab]
    y_pred = [id2lab[i] for i in all_pred]
    swa = shape_weighted_accuracy(all_seq, y_true, y_pred)
    return avg_loss, swa, y_true, y_pred


# ----------- EXPERIMENT TRACKER -----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ----------- TRAINING -----------
model = NeuroSymbolicSPR(len(tok2id), 32, 64, len(labels), pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 25
patience = 3
best_val = -1
no_imp = 0

for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_swa, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_swa, _, _ = run_epoch(model, dev_loader, criterion)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  SWA = {val_swa:.4f}")
    if val_swa > best_val:
        best_val = val_swa
        no_imp = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        no_imp += 1
    if no_imp >= patience:
        print("Early stopping.")
        break

# ----------- TEST -----------
model.load_state_dict(best_state)
model.to(device)
test_loss, test_swa, y_true, y_pred = run_epoch(model, test_loader, criterion)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
experiment_data["SPR_BENCH"]["losses"]["test"] = test_loss
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_swa

# ----------- SAVE DATA & PLOT -----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"])
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(working_dir, "val_loss_curve.png"))
plt.close()
print("Artifacts saved to", working_dir)
