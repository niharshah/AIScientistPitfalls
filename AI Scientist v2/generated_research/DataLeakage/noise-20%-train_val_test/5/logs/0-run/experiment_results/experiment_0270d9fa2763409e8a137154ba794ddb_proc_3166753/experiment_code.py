import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from collections import Counter

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- experiment data dict -------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------- DATA ----------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _ld("train.csv")
    d["dev"] = _ld("dev.csv")
    d["test"] = _ld("test.csv")
    return d


def build_synthetic(n_tr=1000, n_dev=200, n_te=400, L=12, V=8):
    symbols = [chr(ord("A") + i) for i in range(V)]

    def gen(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(L)]
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(int(seq.count("A") % 2 == 0))
        return Dataset.from_dict(data)

    return DatasetDict(train=gen(n_tr), dev=gen(n_dev), test=gen(n_te))


def build_vocab(dset, field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dset[field]:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode(seq, vocab, max_len=None):
    toks = [vocab.get(t, vocab["<unk>"]) for t in seq.split()]
    if max_len:
        toks = toks[:max_len]
    return toks


# ------------------ COLLATE FN ------------------------------
def collate(batch, vocab, max_len=128):
    seqs = [encode(b["sequence"], vocab, max_len) for b in batch]
    lens = [len(s) for s in seqs]
    mx = max(lens)
    pad_id = vocab["<pad>"]
    padded = [s + [pad_id] * (mx - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_id
    # count-vectors
    cnt_dim = len(vocab)
    cnt_mat = np.zeros((len(batch), cnt_dim), dtype=np.float32)
    for i, seq in enumerate(batch):
        c = Counter(seq["sequence"].split())
        for tok, ct in c.items():
            cnt_mat[i, vocab.get(tok, 1)] = ct
    cnt = torch.tensor(cnt_mat, dtype=torch.float32)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"input_ids": x, "attention_mask": mask, "counts": cnt, "labels": y}


# ------------------ MODEL -----------------------------------
class HybridTransformer(nn.Module):
    def __init__(
        self,
        vocab_sz,
        embed_dim,
        nhead,
        nlayers,
        num_cls,
        pad_idx,
        cnt_dim,
        cnt_proj=64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cnt_proj = nn.Linear(cnt_dim, cnt_proj)
        self.classifier = nn.Linear(embed_dim + cnt_proj, num_cls)

    def forward(self, ids, mask, counts):
        pos_idx = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.embed(ids) + self.pos(pos_idx)
        h = self.encoder(h, src_key_padding_mask=mask)
        pooled = (~mask).unsqueeze(-1) * h
        pooled = pooled.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        cnt_feat = torch.relu(self.cnt_proj(counts))
        out = torch.cat([pooled, cnt_feat], dim=-1)
        return self.classifier(out)


# --------------- LOAD dataset --------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic data.", e)
    dsets = build_synthetic()

vocab = build_vocab(dsets["train"])
num_cls = len(set(dsets["train"]["label"]))
print(f"Vocab size {len(vocab)}, num classes {num_cls}")

bs = 64
train_dl = DataLoader(
    dsets["train"], batch_size=bs, shuffle=True, collate_fn=lambda b: collate(b, vocab)
)
dev_dl = DataLoader(
    dsets["dev"], batch_size=bs, shuffle=False, collate_fn=lambda b: collate(b, vocab)
)
test_dl = DataLoader(
    dsets["test"], batch_size=bs, shuffle=False, collate_fn=lambda b: collate(b, vocab)
)


# ---------------- training utils ----------------------------
def macro_f1(preds, labels, n_cls):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    f1s = []
    for c in range(n_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds_all.append(logits.argmax(-1))
            labels_all.append(batch["labels"])
    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    loss = tot_loss / labels.size(0)
    acc = (preds == labels).float().mean().item()
    f1 = macro_f1(preds, labels, num_cls)
    return loss, acc, f1, preds.cpu(), labels.cpu()


# ------------------- TRAIN ----------------------------------
embed_dim = 128
nhead = 2
layers = 2
epochs = 5
model = HybridTransformer(
    len(vocab),
    embed_dim,
    nhead,
    layers,
    num_cls,
    pad_idx=vocab["<pad>"],
    cnt_dim=len(vocab),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for ep in range(1, epochs + 1):
    model.train()
    tot_loss = 0
    correct = 0
    n = 0
    for batch in train_dl:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        opt.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        opt.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        correct += (logits.argmax(-1) == batch["labels"]).sum().item()
        n += batch["labels"].size(0)
    tr_loss = tot_loss / n
    tr_acc = correct / n
    val_loss, val_acc, val_f1, _, _ = evaluate(model, dev_dl, criterion)
    print(
        f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.4f} MacroF1={val_f1:.4f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"epoch": ep, "acc": tr_acc, "f1": None}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": ep, "acc": val_acc, "f1": val_f1}
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

# ------------------- TEST EVAL ------------------------------
test_loss, test_acc, test_f1, preds, labels = evaluate(model, test_dl, criterion)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f} MacroF1={test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = labels.tolist()

# ------------------- SAVE -----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
