import os, pathlib, math, time, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ---------------- housekeeping & GPU ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------- experiment log scaffold ------------
experiment_data = {
    "BagOfEmbeddings": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "SWA": {"val": [], "test": None},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_rec = experiment_data["BagOfEmbeddings"]["SPR_BENCH"]


# --------------- dataset helpers --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
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


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# --------------- load data --------------------------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# --------------- vocabularies -----------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_char_vocab(char_iter):
    vocab = {PAD: 0, UNK: 1}
    for ch in sorted(set(char_iter)):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


shape_chars = (
    tok[0]
    for tok in itertools.chain.from_iterable(
        seq.split() for seq in spr["train"]["sequence"]
    )
)
color_chars = (
    tok[1] if len(tok) > 1 else "_"
    for tok in itertools.chain.from_iterable(
        seq.split() for seq in spr["train"]["sequence"]
    )
)
shape_vocab = build_char_vocab(shape_chars)
color_vocab = build_char_vocab(color_chars)
print(f"Shape vocab {len(shape_vocab)}, Color vocab {len(color_vocab)}")

label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}


# --------------- encoding utilities -----------------
def encode_seq(seq):
    shape_ids = [shape_vocab.get(tok[0], shape_vocab[UNK]) for tok in seq.split()]
    color_ids = [
        color_vocab.get(tok[1] if len(tok) > 1 else "_", color_vocab[UNK])
        for tok in seq.split()
    ]
    return shape_ids, color_ids


# --------------- torch dataset ----------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sids, cids = encode_seq(self.seqs[idx])
        feats = torch.tensor(
            [
                count_shape_variety(self.seqs[idx]),
                len(set(tok[1] for tok in self.seqs[idx].split() if len(tok) > 1)),
                len(self.seqs[idx].split()),
            ],
            dtype=torch.float32,
        )
        return {
            "shape_ids": torch.tensor(sids, dtype=torch.long),
            "color_ids": torch.tensor(cids, dtype=torch.long),
            "sym_feats": feats,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    shp = [b["shape_ids"] for b in batch]
    col = [b["color_ids"] for b in batch]
    feats = torch.stack([b["sym_feats"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    shp_pad = nn.utils.rnn.pad_sequence(
        shp, batch_first=True, padding_value=shape_vocab[PAD]
    )
    col_pad = nn.utils.rnn.pad_sequence(
        col, batch_first=True, padding_value=color_vocab[PAD]
    )
    return {
        "shape_ids": shp_pad,
        "color_ids": col_pad,
        "sym_feats": feats,
        "labels": labels,
        "raw_seq": raw,
    }


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr["train"]),
    SPRTorchDataset(spr["dev"]),
    SPRTorchDataset(spr["test"]),
)
train_loader = DataLoader(train_ds, 128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, shuffle=False, collate_fn=collate)


# --------------- Bag-of-Embeddings model ------------
class BagOfEmbeddings(nn.Module):
    def __init__(self, s_vocab, c_vocab, num_labels, d_model=64):
        super().__init__()
        self.shape_emb = nn.Embedding(s_vocab, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(c_vocab, d_model, padding_idx=0)
        self.feat_norm = nn.LayerNorm(3)
        self.fc = nn.Sequential(nn.Linear(d_model + 3, num_labels))

    def forward(self, shape_ids, color_ids, sym_feats):
        emb = self.shape_emb(shape_ids) + self.color_emb(color_ids)
        mask = shape_ids.eq(0)
        emb = emb.masked_fill(mask.unsqueeze(-1), 0)
        denom = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = emb.sum(dim=1) / denom
        sym_norm = self.feat_norm(sym_feats)
        logits = self.fc(torch.cat([pooled, sym_norm], dim=-1))
        return logits


model = BagOfEmbeddings(len(shape_vocab), len(color_vocab), len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --------------- evaluation -------------------------
def evaluate(loader):
    model.eval()
    tot_loss, total = 0.0, 0
    all_seq, y_true, y_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(
                batch_t["shape_ids"], batch_t["color_ids"], batch_t["sym_feats"]
            )
            loss = criterion(logits, batch_t["labels"])
            pred = logits.argmax(-1)
            bs = batch_t["labels"].size(0)
            tot_loss += loss.item() * bs
            total += bs
            all_seq.extend(batch["raw_seq"])
            y_true.extend(batch_t["labels"].cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    return (
        tot_loss / total,
        shape_weighted_accuracy(all_seq, y_true, y_pred),
        y_pred,
        y_true,
        all_seq,
    )


# --------------- training loop ----------------------
MAX_EPOCHS, PATIENCE = 20, 3
best_val_loss, no_imp = math.inf, 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        batch_t = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch_t["shape_ids"], batch_t["color_ids"], batch_t["sym_feats"])
        loss = criterion(logits, batch_t["labels"])
        loss.backward()
        optimizer.step()
        running += loss.item() * batch_t["labels"].size(0)
    train_loss = running / len(train_ds)

    val_loss, val_swa, *_ = evaluate(dev_loader)
    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  SWA={val_swa:.3f}"
    )

    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["SWA"]["val"].append(val_swa)
    exp_rec["timestamps"].append(time.time())

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print("Early stopping.")
            break

# load best
model.load_state_dict(best_state)

# --------------- final test -------------------------
test_loss, test_swa, preds, trues, seqs = evaluate(test_loader)
print(f"\nTEST: loss={test_loss:.4f}  SWA={test_swa:.3f}")
exp_rec["SWA"]["test"] = test_swa
exp_rec["metrics"]["test"] = {"loss": test_loss, "SWA": test_swa}
exp_rec["predictions"], exp_rec["ground_truth"] = preds, trues

# --------------- save artefacts ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")

plt.figure(figsize=(6, 4))
plt.plot(exp_rec["SWA"]["val"], label="Val SWA")
plt.xlabel("Epoch")
plt.ylabel("SWA")
plt.title("Validation SWA (Bag-of-Embeddings)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "swa_curve.png"))
print("Saved plot to working/swa_curve.png")
