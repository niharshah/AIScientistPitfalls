# ----------------------------------------------------
# Multi-Dataset Generalisation Ablation
# ----------------------------------------------------
import os, pathlib, math, time, itertools, numpy as np, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ---------------- housekeeping -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# --------------- experiment log ---------------------
experiment_data = {
    "MultiDataset": {
        "SPR_BENCH_HELDOUT": {
            "metrics": {"train": [], "val": [], "test": {}},
            "losses": {"train": [], "val": []},
            "SWA": {"val": [], "test": None},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}
exp_rec = experiment_data["MultiDataset"]["SPR_BENCH_HELDOUT"]

# --------------- load original SPR_BENCH ------------
DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)


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


spr = load_spr_bench(DATA_PATH)
print("Original sizes:", {k: len(v) for k, v in spr.items()})

# --------------- utilities --------------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_char_vocab(chars_iter):
    vocab = {PAD: 0, UNK: 1}
    for ch in sorted(set(chars_iter)):
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
shape_vocab, color_vocab = build_char_vocab(shape_chars), build_char_vocab(color_chars)
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def encode_seq(seq):
    shape_ids = [shape_vocab.get(tok[0], shape_vocab[UNK]) for tok in seq.split()]
    color_ids = [
        color_vocab.get(tok[1] if len(tok) > 1 else "_", color_vocab[UNK])
        for tok in seq.split()
    ]
    return shape_ids, color_ids


# --------------- torch dataset ----------------------
class SPRTorchDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seqs, self.labels = sequences, labels

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
            "shape_ids": torch.tensor(sids),
            "color_ids": torch.tensor(cids),
            "sym_feats": feats,
            "label": torch.tensor(label2idx[self.labels[idx]]),
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


# --------------- variant creation -------------------
def create_variant(seed: int, bias: str):
    random.seed(seed)
    np.random.seed(seed)
    seqs, labels = spr["train"]["sequence"], spr["train"]["label"]
    if bias == "shape_A":
        fav = [i for i, s in enumerate(seqs) if s.split()[0][0] == "A"]
    elif bias == "short":
        fav = [i for i, s in enumerate(seqs) if len(s.split()) <= 6]
    elif bias == "long":
        fav = [i for i, s in enumerate(seqs) if len(s.split()) >= 10]
    else:
        fav = list(range(len(seqs)))
    prob = np.ones(len(seqs))
    prob[fav] *= 5.0  # up-weight favourites
    prob /= prob.sum()
    idxs = np.random.choice(len(seqs), size=len(seqs), replace=True, p=prob)
    return [seqs[i] for i in idxs], [labels[i] for i in idxs]


variant_specs = [(111, "shape_A"), (222, "short"), (333, "long")]
variant_datasets = []
for seed, bias in variant_specs:
    v_seqs, v_labels = create_variant(seed, bias)
    variant_datasets.append(SPRTorchDataset(v_seqs, v_labels))
print("Built {} variants.".format(len(variant_datasets)))

# --------------- loaders ----------------------------
train_dataset = ConcatDataset(variant_datasets)
train_loader = DataLoader(train_dataset, 128, shuffle=True, collate_fn=collate)

dev_ds = SPRTorchDataset(spr["dev"]["sequence"], spr["dev"]["label"])
test_ds = SPRTorchDataset(spr["test"]["sequence"], spr["test"]["label"])
dev_loader = DataLoader(dev_ds, 256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, shuffle=False, collate_fn=collate)


# --------------- model ------------------------------
class HybridTransformer(nn.Module):
    def __init__(self, s_vocab, c_vocab, num_labels, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.shape_emb = nn.Embedding(s_vocab, d_model, padding_idx=0)
        self.color_emb = nn.Embedding(c_vocab, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(512, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 128, 0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.feat_norm = nn.LayerNorm(3)
        self.fc = nn.Linear(d_model + 3, num_labels)

    def forward(self, shape_ids, color_ids, sym_feats):
        seq_len = shape_ids.size(1)
        pos = (
            torch.arange(seq_len, device=shape_ids.device)
            .unsqueeze(0)
            .expand(shape_ids.size(0), -1)
        )
        x = self.shape_emb(shape_ids) + self.color_emb(color_ids) + self.pos_emb(pos)
        mask = shape_ids.eq(0)
        h = self.encoder(x, src_key_padding_mask=mask)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        logits = self.fc(torch.cat([pooled, self.feat_norm(sym_feats)], dim=-1))
        return logits


model = HybridTransformer(len(shape_vocab), len(color_vocab), len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --------------- evaluation -------------------------
def evaluate(loader):
    model.eval()
    tot_loss, total = 0.0, 0
    all_seq, y_true, y_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(bt["shape_ids"], bt["color_ids"], bt["sym_feats"])
            loss = criterion(logits, bt["labels"])
            pred = logits.argmax(-1)
            bs = bt["labels"].size(0)
            tot_loss += loss.item() * bs
            total += bs
            all_seq.extend(batch["raw_seq"])
            y_true.extend(bt["labels"].cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    return (tot_loss / total, shape_weighted_accuracy(all_seq, y_true, y_pred))


# --------------- training loop ----------------------
MAX_EPOCHS, PATIENCE = 20, 3
best_val_loss, no_imp = float("inf"), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        bt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(bt["shape_ids"], bt["color_ids"], bt["sym_feats"])
        loss = criterion(logits, bt["labels"])
        loss.backward()
        optimizer.step()
        running += loss.item() * bt["labels"].size(0)
    train_loss = running / len(train_dataset)
    val_loss, val_swa = evaluate(dev_loader)
    print(
        f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | SWA {val_swa:.3f}"
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

# --------------- final test -------------------------
model.load_state_dict(best_state)
test_loss, test_swa = evaluate(test_loader)
print(f"\nTEST ‑ loss {test_loss:.4f} | SWA {test_swa:.3f}")
exp_rec["metrics"]["test"] = {"loss": test_loss, "SWA": test_swa}
exp_rec["SWA"]["test"] = test_swa

# --------------- predictions for saving -------------
model.eval()
preds, trues, seqs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        bt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(bt["shape_ids"], bt["color_ids"], bt["sym_feats"]).argmax(-1)
        preds.extend(out.cpu().tolist())
        trues.extend(bt["labels"].cpu().tolist())
        seqs.extend(batch["raw_seq"])
exp_rec["predictions"], exp_rec["ground_truth"] = preds, trues

# --------------- save artefacts ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

plt.figure(figsize=(6, 4))
plt.plot(exp_rec["SWA"]["val"], label="Val SWA")
plt.xlabel("Epoch")
plt.ylabel("SWA")
plt.title("Validation SWA")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "swa_curve.png"))
print("Plot saved.")
