import os, pathlib, random, time, json, math, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- mandatory working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- deterministic behaviour -----------------
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------- helper SPR utils -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
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


def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.strip().split() if t))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / sum(w)


# ----------------- load data --------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")  # fallback
spr = load_spr_bench(DATA_PATH)
print("Data sizes:", {k: len(v) for k, v in spr.items()})


# ----------------- vocabularies ------------------------
def tokens(seq):
    return seq.strip().split()


word2id = {"<pad>": 0, "<unk>": 1}
shape_set, color_set = set(), set()
for ex in spr["train"]:
    for tok in tokens(ex["sequence"]):
        if tok not in word2id:
            word2id[tok] = len(word2id)
        if len(tok) > 0:
            shape_set.add(tok[0])
        if len(tok) > 1:
            color_set.add(tok[1])

shape2idx = {s: i for i, s in enumerate(sorted(shape_set))}
color2idx = {c: i for i, c in enumerate(sorted(color_set))}
vocab_size = len(word2id)
print("Vocab:", vocab_size, "Shapes:", len(shape2idx), "Colors:", len(color2idx))

label_set = sorted({ex["label"] for ex in spr["train"]})
lab2id = {l: i for i, l in enumerate(label_set)}
num_classes = len(lab2id)
print("Classes:", num_classes)


# ----------------- dataset class ------------------------
class SPRDataset(Dataset):
    def __init__(self, split, word2id, lab2id, shape2idx, color2idx):
        self.data = split
        self.w2i, self.l2i = word2id, lab2id
        self.shape2i, self.color2i = shape2idx, color2idx
        self.sh_dim = len(shape2idx)
        self.co_dim = len(color2idx)

    def __len__(self):
        return len(self.data)

    def _sym_feats(self, seq):
        sh = np.zeros(self.sh_dim, dtype=np.float32)
        co = np.zeros(self.co_dim, dtype=np.float32)
        toks = tokens(seq)
        for tok in toks:
            if tok:
                if tok[0] in self.shape2i:
                    sh[self.shape2i[tok[0]]] += 1.0
                if len(tok) > 1 and tok[1] in self.color2i:
                    co[self.color2i[tok[1]]] += 1.0
        var_shape = float(count_shape_variety(seq))
        length = float(len(toks))
        sym = np.concatenate([sh, co, [var_shape, length]], axis=0)
        return sym

    def __getitem__(self, idx):
        row = self.data[idx]
        tok_ids = [self.w2i.get(t, self.w2i["<unk>"]) for t in tokens(row["sequence"])]
        sym = self._sym_feats(row["sequence"])
        return {
            "ids": torch.tensor(tok_ids, dtype=torch.long),
            "sym": torch.tensor(sym, dtype=torch.float32),
            "label": torch.tensor(self.l2i[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


def collate_fn(batch):
    ids = [b["ids"] for b in batch]
    lens = [len(x) for x in ids]
    pad_ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    sym = torch.stack([b["sym"] for b in batch])
    lbl = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "ids": pad_ids,
        "lengths": torch.tensor(lens),
        "sym": sym,
        "label": lbl,
        "raw": raw,
    }


bs = 256
train_loader = DataLoader(
    SPRDataset(spr["train"], word2id, lab2id, shape2idx, color2idx),
    batch_size=bs,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"], word2id, lab2id, shape2idx, color2idx),
    batch_size=bs,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    SPRDataset(spr["test"], word2id, lab2id, shape2idx, color2idx),
    batch_size=bs,
    shuffle=False,
    collate_fn=collate_fn,
)


# ----------------- model -----------------------------
class NeuralSymbolicClassifier(nn.Module):
    def __init__(self, vocab, sym_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden_dim, padding_idx=0)
        self.fc = nn.Linear(hidden_dim + sym_dim, num_classes)

    def forward(self, ids, sym):
        emb = self.emb(ids)  # B x T x H
        mask = (ids != 0).unsqueeze(-1)
        avg = (emb * mask).sum(1) / (mask.sum(1).clamp(min=1))
        x = torch.cat([avg, sym], dim=-1)
        return self.fc(x)


sym_dim = len(shape2idx) + len(color2idx) + 2
model = NeuralSymbolicClassifier(vocab_size, sym_dim, 64, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------- experiment store -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": spr["dev"]["label"],
    }
}
epochs = 6


# ----------------- evaluation helper ----------------
def evaluate(loader):
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    preds, raws, trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            sym = batch["sym"].to(device)
            lab = batch["label"].to(device)
            logits = model(ids, sym)
            loss = criterion(logits, lab)
            tot_loss += loss.item() * lab.size(0)
            p = logits.argmax(-1)
            correct += (p == lab).sum().item()
            n += lab.size(0)
            preds.extend(p.cpu().tolist())
            raws.extend(batch["raw"])
            trues.extend(lab.cpu().tolist())
    acc = correct / n
    swa = shape_weighted_accuracy(raws, trues, preds)
    return tot_loss / n, acc, swa, preds, raws, trues


# ----------------- training loop --------------------
for epoch in range(1, epochs + 1):
    model.train()
    run_loss, correct, n = 0.0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch["ids"].to(device)
        sym = batch["sym"].to(device)
        lab = batch["label"].to(device)
        logits = model(ids, sym)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * lab.size(0)
        p = logits.argmax(-1)
        correct += (p == lab).sum().item()
        n += lab.size(0)
    train_loss = run_loss / n
    train_acc = correct / n
    val_loss, val_acc, val_swa, _, _, _ = evaluate(dev_loader)

    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["train_swa"].append(None)  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(val_swa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.3f} | val_SWA={val_swa:.3f}"
    )

# ----------------- final test evaluation -------------
test_loss, test_acc, test_swa, preds, raws, trues = evaluate(test_loader)
print(f"\nTEST: CE={test_loss:.4f}  acc={test_acc:.3f}  SWA={test_swa:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
