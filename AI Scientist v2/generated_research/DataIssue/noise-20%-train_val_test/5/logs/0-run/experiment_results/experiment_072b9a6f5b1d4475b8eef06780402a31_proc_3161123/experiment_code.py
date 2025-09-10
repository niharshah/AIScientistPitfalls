import os, pathlib, random, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# -----------------------  experiment dict & dirs -------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "num_layers": {"SPR_BENCH": {}}  # hyper-parameter tuning type  # dataset name
}

# ------------------------------ device ------------------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------ data -------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for s in ["train", "dev", "test"]:
        dd[s] = _ld(f"{s}.csv")
    return dd


def build_synthetic(ntr=500, ndv=100, nte=200, seqlen=10, vsz=12):
    syms = [chr(ord("A") + i) for i in range(vsz)]

    def gen(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(syms) for _ in range(seqlen)]
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(1 if seq.count("A") % 2 == 0 else 0)
        return Dataset.from_dict(d)

    return DatasetDict(train=gen(ntr), dev=gen(ndv), test=gen(nte))


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Falling back to synthetic data:", e)
    datasets_dict = build_synthetic()


# --------------------------- vocab & utils -------------------------- #
def build_vocab(ds, field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in ds[field]:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab(datasets_dict["train"])
pad_idx = vocab["<pad>"]


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(t, vocab["<unk>"]) for t in seq.split()]
    return toks[:max_len] if max_len else toks


def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    L = max(len(s) for s in seqs)
    padded = [s + [pad_idx] * (L - len(s)) for s in seqs]
    x = torch.tensor(padded)
    mask = x == pad_idx
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


batch_size = 64
train_dl = DataLoader(
    datasets_dict["train"], batch_size, True, collate_fn=lambda b: collate_fn(b, vocab)
)
dev_dl = DataLoader(
    datasets_dict["dev"], batch_size, False, collate_fn=lambda b: collate_fn(b, vocab)
)
test_dl = DataLoader(
    datasets_dict["test"], batch_size, False, collate_fn=lambda b: collate_fn(b, vocab)
)


# -------------------------- model classes --------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            embed_dim,
            nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        pos_idx = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_idx)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_flt = (~mask).unsqueeze(-1)
        h_sum = (h * mask_flt).sum(1)
        lengths = mask_flt.sum(1).clamp(min=1)
        pooled = h_sum / lengths
        return self.cls(pooled)


# ------------------------- evaluation ------------------------------- #
def evaluate(model, loader, criterion):
    model.eval()
    loss_tot, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(b["input_ids"], b["attention_mask"])
            loss = criterion(logits, b["labels"])
            loss_tot += loss.item() * b["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == b["labels"]).sum().item()
            count += b["labels"].size(0)
    return loss_tot / count, correct / count


# ---------------------- training for each depth --------------------- #
embed_dim, nhead = 128, 4
num_classes = len(set(datasets_dict["train"]["label"]))
epochs = 5
candidate_layers = [1, 2, 3, 4]

for L in candidate_layers:
    print(f"\n=== Training with num_layers={L} ===")
    model = SimpleTransformerClassifier(
        len(vocab), embed_dim, nhead, L, num_classes, pad_idx
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    m_train_acc, m_val_acc, m_train_loss, m_val_loss = [], [], [], []

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, correct, count = 0.0, 0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            count += batch["labels"].size(0)
        train_loss = tot_loss / count
        train_acc = correct / count
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        m_train_loss.append(train_loss)
        m_val_loss.append(val_loss)
        m_train_acc.append(train_acc)
        m_val_acc.append(val_acc)
        print(f"Epoch {ep}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    # ---------- test evaluation ----------
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"Test accuracy for {L} layers: {test_acc:.3f}")

    # ---------- predictions --------------
    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            gpu = {k: v.to(device) for k, v in batch.items()}
            logits = model(gpu["input_ids"], gpu["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            gts_all.extend(batch["labels"].tolist())

    # ---------- log to experiment_data ----
    key = f"layers_{L}"
    experiment_data["num_layers"]["SPR_BENCH"][key] = {
        "metrics": {"train_acc": m_train_acc, "val_acc": m_val_acc},
        "losses": {"train": m_train_loss, "val": m_val_loss},
        "test_acc": test_acc,
        "predictions": preds_all,
        "ground_truth": gts_all,
    }

# ------------------------ save results ------------------------------ #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
