import os, pathlib, time, math, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------------------------------------- house-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------- experiment log
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
log = experiment_data["SPR_BENCH"]


# -------------------------------------------------- helper funcs
def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, ys_true, ys_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, ys_true, ys_pred)) / max(
        sum(w), 1
    )


# -------------------------------------------------- data loading
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


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
dset = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dset.items()})

# -------------------------------------------------- vocab
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(split):
    vocab = {PAD: 0, UNK: 1}
    tokens = set(itertools.chain.from_iterable(s.strip().split() for s in split))
    for tok in sorted(tokens):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(dset["train"]["sequence"])


def encode(seq: str):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()]


labels = sorted(set(dset["train"]["label"]))
lab2idx = {l: i for i, l in enumerate(labels)}
idx2lab = {i: l for l, i in lab2idx.items()}


# -------------------------------------------------- torch dataset
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [lab2idx[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "ids": torch.tensor(encode(seq), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": seq,
            "shape_cnt": torch.tensor(count_shape_variety(seq), dtype=torch.float),
            "color_cnt": torch.tensor(count_color_variety(seq), dtype=torch.float),
        }


def collate(batch):
    ids = [b["ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    shapes = torch.stack([b["shape_cnt"] for b in batch])
    colors = torch.stack([b["color_cnt"] for b in batch])
    raw_seqs = [b["raw"] for b in batch]
    pad_ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=vocab[PAD])
    return {
        "input_ids": pad_ids,
        "labels": labels,
        "shape_cnt": shapes,
        "color_cnt": colors,
        "raw_seq": raw_seqs,
    }


train_ds, dev_ds, test_ds = (
    SPRTorch(dset["train"]),
    SPRTorch(dset["dev"]),
    SPRTorch(dset["test"]),
)
train_loader = DataLoader(train_ds, 128, True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, False, collate_fn=collate)


# -------------------------------------------------- model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab_size, n_labels, d_model=64, nhead=2, nlayers=2, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.fc = nn.Linear(d_model + 2, n_labels)  # +2 for symbolic counts

    def forward(self, ids, shape_cnt, color_cnt):
        mask = ids == 0
        x = self.pos(self.emb(ids))
        h = self.encoder(x, src_key_padding_mask=mask)
        h_mean = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1).clamp(
            min=1
        ).unsqueeze(-1)
        sym = torch.stack([shape_cnt, color_cnt], dim=1) / 10.0  # crude normalisation
        logits = self.fc(torch.cat([h_mean, sym], dim=1))
        return logits


model = TransformerClassifier(len(vocab), len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# -------------------------------------------------- evaluation
def evaluate(loader):
    model.eval()
    tot_loss, total = 0.0, 0
    all_pred, all_true, all_seq = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            shp = batch["shape_cnt"].to(device)
            col = batch["color_cnt"].to(device)
            logits = model(ids, shp, col)
            loss = criterion(logits, lab)
            tot_loss += loss.item() * lab.size(0)
            pred = logits.argmax(-1)
            total += lab.size(0)
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(lab.cpu().tolist())
            all_seq.extend(batch["raw_seq"])
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    return tot_loss / total, swa, all_pred, all_true, all_seq


# -------------------------------------------------- training
MAX_EPOCHS, PATIENCE = 15, 3
best_val, wait = float("inf"), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)
        shp = batch["shape_cnt"].to(device)
        col = batch["color_cnt"].to(device)
        optimizer.zero_grad()
        logits = model(ids, shp, col)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * lab.size(0)
    train_loss = epoch_loss / len(train_ds)
    val_loss, val_swa, *_ = evaluate(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  SWA = {val_swa:.3f}")
    log["losses"]["train"].append(train_loss)
    log["losses"]["val"].append(val_loss)
    log["metrics"]["val"].append({"epoch": epoch, "swa": val_swa})
    log["timestamps"].append(time.time())
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping after {epoch} epochs")
            break

model.load_state_dict(best_state)

# -------------------------------------------------- final test
test_loss, test_swa, preds, trues, seqs = evaluate(test_loader)
print(f"\nTEST  loss={test_loss:.4f}  SWA={test_swa:.3f}")
log["metrics"]["test"] = {"loss": test_loss, "swa": test_swa}
log["predictions"], log["ground_truth"] = preds, trues

# -------------------------------------------------- save / plot
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(["SWA"], [test_swa], color="coral")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Shape-Weighted Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "swa_bar.png"))
print(f"Data & plot saved in {working_dir}")
