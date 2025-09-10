import os, pathlib, random, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict

# ---------- EXPERIMENT & DEVICE SET-UP ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- DATA LOADING ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_synthetic(n_train=800, n_dev=200, n_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(int(seq.count("A") % 2 == 0))
        return Dataset.from_dict(d)

    return DatasetDict(train=gen(n_train), dev=gen(n_dev), test=gen(n_test))


try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH")
except Exception as e:
    print("Falling back to synthetic data:", e)
    dsets = build_synthetic()

# ---------- VOCAB ----------
SPECIALS = ["<pad>", "<unk>", "[CLS]"]


def build_vocab(dataset):
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    next_idx = len(SPECIALS)
    for s in dataset["sequence"]:
        for t in s.strip().split():
            if t not in vocab:
                vocab[t] = next_idx
                next_idx += 1
    return vocab


vocab = build_vocab(dsets["train"])
pad_id = vocab["<pad>"]
cls_id = vocab["[CLS]"]
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab size={len(vocab)}, classes={num_classes}")


# ---------- COLLATE ----------
def encode(seq, vocab, max_len=None):
    ids = [vocab.get(t, vocab["<unk>"]) for t in seq.strip().split()]
    if max_len:
        ids = ids[:max_len]
    return ids


def collate(batch, vocab, max_len=128):
    toks = [[cls_id] + encode(b["sequence"], vocab, max_len) for b in batch]
    lens = [len(t) for t in toks]
    max_l = max(lens)
    padded = [t + [pad_id] * (max_l - len(t)) for t in toks]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_id
    # global count features (excluding specials)
    counts = np.zeros((len(batch), len(vocab)), dtype=np.float32)
    for i, tok_seq in enumerate(toks):
        for tok in tok_seq[1:]:  # exclude CLS
            counts[i, tok] += 1
    counts = torch.tensor(counts)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "input_ids": x,
        "attention_mask": mask,
        "global_feats": counts,
        "labels": labels,
    }


batch_size = 64
train_loader = DataLoader(
    dsets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate(b, vocab),
)
dev_loader = DataLoader(
    dsets["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)
test_loader = DataLoader(
    dsets["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)


# ---------- MODEL ----------
class HybridClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        nhead=2,
        num_layers=2,
        num_classes=2,
        pad_idx=0,
        global_dim=128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.global_proj = nn.Sequential(
            nn.Linear(vocab_size, global_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(embed_dim + global_dim, num_classes)

    def forward(self, ids, mask, g_feats):
        pos_ids = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.embed(ids) + self.pos(pos_ids)
        h_enc = self.encoder(h, src_key_padding_mask=mask)
        cls_vec = h_enc[:, 0]  # first token ([CLS])
        g_vec = self.global_proj(g_feats)
        logits = self.classifier(torch.cat([cls_vec, g_vec], dim=-1))
        return logits


# ---------- METRIC ----------
def macro_f1(preds, labels, num_classes):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp == 0 and (fp == 0 or fn == 0):
            f1 = 0.0
        else:
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = tot = correct = 0
    all_p = []
    all_l = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["global_feats"]
            )
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            tot += batch["labels"].size(0)
            all_p.append(preds)
            all_l.append(batch["labels"])
    preds = torch.cat(all_p)
    labels = torch.cat(all_l)
    acc = correct / tot
    f1 = macro_f1(preds, labels, num_classes)
    return tot_loss / tot, acc, f1, preds.cpu(), labels.cpu()


# ---------- TRAINING ----------
embed_dim = 128
nhead = 2
epochs = 5
model = HybridClassifier(len(vocab), embed_dim, nhead, 2, num_classes, pad_id).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = correct = tot = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"], batch["attention_mask"], batch["global_feats"]
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        tot += batch["labels"].size(0)
    train_loss = epoch_loss / tot
    train_acc = correct / tot
    _, _, train_f1, _, _ = evaluate(model, train_loader, criterion)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, dev_loader, criterion)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    print(
        f"  Train  acc={train_acc:.4f} f1={train_f1:.4f} | "
        f"Val acc={val_acc:.4f} f1={val_f1:.4f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

# ---------- TEST EVALUATION ----------
test_loss, test_acc, test_f1, preds, gts = evaluate(model, test_loader, criterion)
print(f"Test accuracy={test_acc:.4f}  macroF1={test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = gts.tolist()

# ---------- SAVE ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
