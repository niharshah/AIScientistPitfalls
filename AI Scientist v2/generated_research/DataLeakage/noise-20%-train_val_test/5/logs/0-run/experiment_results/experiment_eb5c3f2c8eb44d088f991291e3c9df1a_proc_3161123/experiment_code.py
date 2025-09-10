import os, pathlib, random, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ------------------------ I/O & STORAGE ----------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "transformer_dropout_rate": {
        "SPR_BENCH": {}  # will be filled with per-dropout results
    }
}

# --------------------------- DEVICE --------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- DATA ----------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def build_synthetic(ntr=500, ndev=100, nte=200, seqlen=10, vocab_sz=12):
    syms = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen(n):
        dat = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(syms) for _ in range(seqlen)]
            lab = 1 if seq.count("A") % 2 == 0 else 0
            dat["id"].append(str(i))
            dat["sequence"].append(" ".join(seq))
            dat["label"].append(lab)
        return Dataset.from_dict(dat)

    return DatasetDict(train=gen(ntr), dev=gen(ndev), test=gen(nte))


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH data.")
except Exception as e:
    print("Falling back to synthetic:", e)
    dsets = build_synthetic()


def build_vocab(dataset: Dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()]
    return toks[:max_len] if max_len else toks


def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    maxlen = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (maxlen - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


# --------------------------- MODEL ---------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nlayers, nclass, pad_idx, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.classifier = nn.Linear(embed_dim, nclass)

    def forward(self, x, pad_mask):
        pos_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        mask_flt = (~pad_mask).unsqueeze(-1)
        h_sum = (h * mask_flt).sum(1)
        lens = mask_flt.sum(1).clamp(min=1)
        pooled = h_sum / lens
        return self.classifier(pooled)


def evaluate(model, dl, crit):
    model.eval()
    tot_loss = tot_correct = cnt = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = crit(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            tot_correct += (preds == batch["labels"]).sum().item()
            cnt += batch["labels"].size(0)
    return tot_loss / cnt, tot_correct / cnt


# ------------------------- PREP ------------------------------------- #
vocab = build_vocab(dsets["train"])
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab size={len(vocab)}, num_classes={num_classes}")

dropout_grid = [0.0, 0.1, 0.2, 0.3]
batch_size = 64
train_dl = DataLoader(
    dsets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab),
)
dev_dl = DataLoader(
    dsets["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)
test_dl = DataLoader(
    dsets["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)

# -------------------- GRID SEARCH OVER DROPOUT ---------------------- #
for dp in dropout_grid:
    key = f"dropout_{dp}"
    experiment_data["transformer_dropout_rate"]["SPR_BENCH"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = SimpleTransformerClassifier(
        len(vocab), 128, 4, 2, num_classes, vocab["<pad>"], dropout=dp
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for ep in range(1, epochs + 1):
        # training
        model.train()
        ep_loss = cor = tot = 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            cor += (preds == batch["labels"]).sum().item()
            tot += batch["labels"].size(0)
        train_loss = ep_loss / tot
        train_acc = cor / tot

        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        print(f"dp={dp} | epoch {ep}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        ed = experiment_data["transformer_dropout_rate"]["SPR_BENCH"][key]
        ed["metrics"]["train"].append(train_acc)
        ed["metrics"]["val"].append(val_acc)
        ed["losses"]["train"].append(train_loss)
        ed["losses"]["val"].append(val_loss)

    # test evaluation
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"dp={dp} | TEST ACCURACY={test_acc:.3f}")
    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            gts_all.extend(batch["labels"].tolist())
    ed["predictions"] = preds_all
    ed["ground_truth"] = gts_all

# ------------------------- SAVE ------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment results to", os.path.join(working_dir, "experiment_data.npy"))
