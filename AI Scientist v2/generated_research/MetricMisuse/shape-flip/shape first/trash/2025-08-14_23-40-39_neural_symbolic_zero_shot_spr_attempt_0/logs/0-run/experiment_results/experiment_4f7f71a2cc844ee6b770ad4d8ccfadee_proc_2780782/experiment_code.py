import os, pathlib, math, numpy as np, torch
from collections import Counter
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, disable_caching

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment recorder ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_swa": []},
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
        "timestamps": [],
    }
}

# ---------- hyper-params ----------
EMB_DIM = 128
HID_DIM = 256
BATCH = 128
EPOCHS = 30
LR = 3e-4
PATIENCE = 4
PAD, UNK = "<pad>", "<unk>"


# ---------- utils ----------
def spr_path() -> pathlib.Path:
    env = os.getenv("SPR_PATH")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    fallback = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fallback / "train.csv").exists():
        return fallback
    raise FileNotFoundError("SPR_BENCH not found; set SPR_PATH")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv",
            data_files=str(root / name),
            split="train",
            cache_dir=str(working_dir) + "/.cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------- data --------------
disable_caching()
DATA_PATH = spr_path()
spr = load_spr_bench(DATA_PATH)
train_sequences = spr["train"]["sequence"]
token_counter = Counter(tok for seq in train_sequences for tok in seq.split())
vocab = {PAD: 0, UNK: 1}
for tok in token_counter:
    vocab.setdefault(tok, len(vocab))
inv_vocab = {i: t for t, i in vocab.items()}

label_set = sorted(set(spr["train"]["label"]))
lbl2id = {l: i for i, l in enumerate(label_set)}
id2lbl = {i: l for l, i in lbl2id.items()}


def encode(seq: str):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode(seq), dtype=torch.long),
            "sym_feats": torch.tensor(
                [
                    count_shape_variety(seq),
                    len(set(tok[1] for tok in seq.split() if len(tok) > 1)),
                ],
                dtype=torch.float,
            ),
            "labels": torch.tensor(lbl2id[self.labels[idx]], dtype=torch.long),
            "seq_str": seq,
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
    return {
        "input_ids": ids,
        "sym_feats": torch.stack([b["sym_feats"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "seq_strs": [b["seq_str"] for b in batch],
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH, shuffle=False, collate_fn=collate
)


# -------------- model --------------
class NeuralSymbolic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim + 2, hid_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, n_classes)

    def forward(self, ids, sym):
        mask = (ids != 0).float().unsqueeze(-1)
        avg = (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        x = torch.cat([avg, sym], dim=1)
        return self.fc2(self.act(self.fc1(x)))


model = NeuralSymbolic(len(vocab), EMB_DIM, HID_DIM, len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)


# -------------- evaluation --------------
def evaluate(loader):
    model.eval()
    tot_loss, n = 0, 0
    preds, gts, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["sym_feats"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            p = logits.argmax(1).cpu().tolist()
            preds.extend(p)
            gts.extend(batch["labels"].cpu().tolist())
            seqs.extend(batch["seq_strs"])
    return tot_loss / max(n, 1), shape_weighted_accuracy(seqs, gts, preds), preds, gts


# -------------- training --------------
best_swa, patience = -1, 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, seen = 0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optim.zero_grad()
        logits = model(batch["input_ids"], batch["sym_feats"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optim.step()
        run_loss += loss.item() * batch["labels"].size(0)
        seen += batch["labels"].size(0)
    train_loss = run_loss / seen

    val_loss, val_swa, _, _ = evaluate(dev_loader)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | SWA={val_swa:.4f}"
    )

    # record
    exp = experiment_data["SPR_BENCH"]["metrics"]
    exp["train_loss"].append(train_loss)
    exp["val_loss"].append(val_loss)
    exp["val_swa"].append(val_swa)
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

# -------------- restore best --------------
if best_state:
    model.load_state_dict(best_state)

# -------------- final eval --------------
dev_loss, dev_swa, dev_pred, dev_gt = evaluate(dev_loader)
test_loss, test_swa, test_pred, test_gt = evaluate(test_loader)

print(f"=== DEV  === loss {dev_loss:.4f} | SWA {dev_swa:.4f}")
print(f"=== TEST === loss {test_loss:.4f} | SWA {test_swa:.4f}")

# save predictions
experiment_data["SPR_BENCH"]["predictions"]["dev"] = dev_pred
experiment_data["SPR_BENCH"]["ground_truth"]["dev"] = dev_gt
experiment_data["SPR_BENCH"]["predictions"]["test"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"]["test"] = test_gt

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
