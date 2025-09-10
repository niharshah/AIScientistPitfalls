import os, pathlib, random, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ----------------- basic set-up --------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------- load SPR_BENCH ------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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


candidates = ["./SPR_BENCH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH"]
for c in candidates:
    if pathlib.Path(c).exists():
        DATA_ROOT = pathlib.Path(c)
        break
else:
    raise FileNotFoundError("SPR_BENCH dataset not found in expected locations")

spr = load_spr_bench(DATA_ROOT)
print({k: len(v) for k, v in spr.items()})


# ----------------- utility metrics -----------------------------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ----------------- vocabulary & symbolic sets ------------------------------
def tokens(seq):
    return seq.strip().split()


vocab = {"<pad>": 0, "<unk>": 1}
shape_set, color_set = set(), set()
for ex in spr["train"]:
    for tok in tokens(ex["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
        shape_set.add(tok[0])
        if len(tok) > 1:
            color_set.add(tok[1])

vocab_size = len(vocab)
shape_list, color_list = sorted(shape_set), sorted(color_set)
shape2i = {s: i for i, s in enumerate(shape_list)}
color2i = {c: i for i, c in enumerate(color_list)}
sym_dim = len(shape_list) + len(color_list)
print(f"Vocab={vocab_size}  Shapes={len(shape_list)}  Colours={len(color_list)}")

labels = sorted({ex["label"] for ex in spr["train"]})
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)


# ----------------- dataset --------------------------------------------------
class SPRSymDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        tok_ids = [vocab.get(t, vocab["<unk>"]) for t in tokens(row["sequence"])]
        # symbolic feature vector
        svec = np.zeros(sym_dim, dtype=np.float32)
        for t in tokens(row["sequence"]):
            svec[shape2i[t[0]]] = 1.0
            if len(t) > 1:
                svec[len(shape_list) + color2i[t[1]]] = 1.0
        return {
            "ids": torch.tensor(tok_ids, dtype=torch.long),
            "sym": torch.tensor(svec, dtype=torch.float32),
            "label": torch.tensor(label2id[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


def collate(batch):
    ids = [b["ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    sym = torch.stack([b["sym"] for b in batch])
    lab = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {"ids": padded, "sym": sym, "label": lab, "raw_seq": raw}


batch_size = 256
train_loader = DataLoader(
    SPRSymDataset(spr["train"]), batch_size, True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRSymDataset(spr["dev"]), batch_size, False, collate_fn=collate
)
test_loader = DataLoader(
    SPRSymDataset(spr["test"]), batch_size, False, collate_fn=collate
)


# ----------------- model ----------------------------------------------------
class NeuralSymbolic(nn.Module):
    def __init__(self, vocab_size, sym_dim, embed_dim=64, sym_h=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.sym = nn.Linear(sym_dim, sym_h)
        self.out = nn.Linear(embed_dim + sym_h, num_classes)

    def forward(self, ids, sym):
        emb = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        sym_h = torch.relu(self.sym(sym))
        x = torch.cat([avg, sym_h], dim=1)
        return self.out(x)


# ----------------- training helpers ----------------------------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optim=None):
    train = optim is not None
    tot_loss, tot_correct, tot_items = 0.0, 0, 0
    seq_col, pred_col, true_col = [], [], []
    for batch in loader:
        ids = batch["ids"].to(device)
        sy = batch["sym"].to(device)
        y = batch["label"].to(device)
        logits = model(ids, sy)
        loss = criterion(logits, y)
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * y.size(0)
        preds = logits.argmax(-1)
        tot_correct += (preds == y).sum().item()
        tot_items += y.size(0)
        seq_col.extend(batch["raw_seq"])
        pred_col.extend(preds.cpu().numpy())
        true_col.extend(y.cpu().numpy())
    acc = tot_correct / tot_items
    swa = shape_weighted_accuracy(seq_col, true_col, pred_col)
    return tot_loss / tot_items, acc, swa, pred_col, true_col, seq_col


# ----------------- training loop -------------------------------------------
lr = 1e-3
epochs = 8
model = NeuralSymbolic(vocab_size, sym_dim).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

for ep in range(1, epochs + 1):
    tr_loss, tr_acc, tr_swa, *_ = run_epoch(model, train_loader, optim)
    vl_loss, vl_acc, vl_swa, vl_pred, vl_true, _ = run_epoch(model, dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vl_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_swa"].append(tr_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(vl_swa)
    print(f"Epoch {ep}: val_loss={vl_loss:.4f}  SWA={vl_swa:.3f}")

# ----------------- final test evaluation -----------------------------------
ts_loss, ts_acc, ts_swa, ts_pred, ts_true, ts_seq = run_epoch(model, test_loader)
print(f"\nTEST  Shape-Weighted Accuracy (SWA) = {ts_swa:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = ts_pred
experiment_data["SPR_BENCH"]["ground_truth"] = ts_true
experiment_data["SPR_BENCH"]["test_swa"] = ts_swa

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
