import os, pathlib, math, numpy as np, torch
from collections import Counter
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ============== working dir ==========================
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ============== device ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============== experiment data container ============
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_swa": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
        "timestamps": [],
    }
}

# ============== Utilities ============================
disable_caching()
PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"


def resolve_spr_path() -> pathlib.Path:
    env = os.getenv("SPR_PATH")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    fb = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fb / "train.csv").exists():
        return fb
    raise FileNotFoundError("Could not find SPR_BENCH (set SPR_PATH).")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv",
            data_files=str(root / csv),
            split="train",
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ============== Load data & vocab =====================
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)

tok_counter = Counter(tok for s in spr["train"]["sequence"] for tok in s.split())
vocab = {PAD: 0, UNK: 1, CLS: 2}
for tok in tok_counter:
    vocab[tok] = len(vocab)
id2tok = {i: t for t, i in vocab.items()}

label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(label2id)
print(f"Vocab {len(vocab)} | Classes {NUM_CLASSES}")


def encode_sequence(seq: str):
    return [vocab[CLS]] + [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return dict(
            input_ids=torch.tensor(encode_sequence(seq), dtype=torch.long),
            labels=torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
            shape_cnt=torch.tensor([count_shape_variety(seq)], dtype=torch.float),
            seq_str=seq,
        )


def collate_fn(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
    return dict(
        input_ids=ids,
        labels=torch.stack([b["labels"] for b in batch]),
        shape_cnt=torch.stack([b["shape_cnt"] for b in batch]),
        seq_strs=[b["seq_str"] for b in batch],
    )


BATCH = 128
train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)

# ============== Model =================================
EMB = 128
HID = 256
NHEAD = 4
NLAYER = 2


class NeuralSymbolicSPR(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_cls):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(512, emb_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=NHEAD, dim_feedforward=HID, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=NLAYER)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim + 1, HID), nn.ReLU(), nn.Linear(HID, num_cls)
        )

    def forward(self, ids, shape_cnt):
        B, L = ids.size()
        pos = self.pos[:L]
        x = self.emb(ids) + pos
        key_padding = ids.eq(0)
        h = self.transformer(x, src_key_padding_mask=key_padding)
        cls = h[:, 0]  # representation of <cls>
        feat = torch.cat([cls, shape_cnt / 10.0], dim=1)  # normalize count
        return self.fc(feat)


model = NeuralSymbolicSPR(len(vocab), EMB, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ============== Train / Eval helpers ==================
def run_epoch(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss, n = 0, 0
    preds, labels, seqs = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["shape_cnt"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = batch["labels"].size(0)
        tot_loss += loss.item() * bs
        n += bs
        p = logits.argmax(1).cpu().tolist()
        l = batch["labels"].cpu().tolist()
        preds.extend(p)
        labels.extend(l)
        seqs.extend(batch["seq_strs"])
    loss = tot_loss / max(n, 1)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    return loss, swa, preds, labels


# ============== Training loop =========================
EPOCHS = 30
PATIENCE = 5
best_swa = -1
patience = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_swa, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_swa, _, _ = run_epoch(dev_loader, train=False)
    print(
        f"Epoch {epoch}: train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_SWA {val_swa:.4f}"
    )
    # log
    m = experiment_data["SPR_BENCH"]["metrics"]
    m["train_loss"].append(tr_loss)
    m["val_loss"].append(val_loss)
    m["val_swa"].append(val_swa)
    experiment_data["SPR_BENCH"]["timestamps"].append(datetime.utcnow().isoformat())
    # early stopping
    if val_swa > best_swa:
        best_swa = val_swa
        best_state = model.state_dict()
        patience = 0
    else:
        patience += 1
    if patience >= PATIENCE:
        print("Early stopping.")
        break

# restore best
if best_state is not None:
    model.load_state_dict(best_state)

# ============== Final evaluation =====================
dev_loss, dev_swa, dev_preds, dev_labels = run_epoch(dev_loader, False)
test_loss, test_swa, test_preds, test_labels = run_epoch(test_loader, False)
print(f"=== DEV  === loss {dev_loss:.4f} | SWA {dev_swa:.4f}")
print(f"=== TEST === loss {test_loss:.4f} | SWA {test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_preds
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_labels
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
