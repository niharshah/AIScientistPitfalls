import os, pathlib, random, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score

# -------------------- HOUSEKEEPING & DEVICE -------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- EXPERIMENT DATA STRUCT ------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train": {"loss": [], "f1": []},
            "val": {"loss": [], "f1": []},
            "test": {},
        },
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------- DATA LOADING ----------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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


def build_vocab(dset):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for seq in dset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode(seq, vocab, max_len=None):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    if max_len:
        ids = ids[:max_len]
    return ids


def collate(batch, vocab, max_len=64):
    seqs = [encode(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    maxL = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (maxL - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x.eq(vocab["<pad>"])
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


# fallback synthetic data if real dataset absent
def synthetic_build(n_train=2000, n_dev=400, n_test=800, seqlen=10):
    symbols = [chr(ord("A") + i) for i in range(12)]

    def make(n, start):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            label = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(start + i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(label)
        return data

    split = {}
    split["train"] = make(n_train, 0)
    split["dev"] = make(n_dev, n_train)
    split["test"] = make(n_test, n_train + n_dev)
    return DatasetDict(
        {
            k: load_dataset("json", data_files={"tmp": [v]}, field="tmp", split="train")
            .shuffle(seed=42)
            .map(lambda ex: ex)
            for k, v in split.items()
        }
    )


try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH")
except Exception as e:
    print("Could not load SPR_BENCH, falling back to synthetic.", e)
    dsets = synthetic_build()

vocab = build_vocab(dsets["train"])
pad_id = vocab["<pad>"]
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab size={len(vocab)}, num_classes={num_classes}")


# -------------------- MODEL ------------------------------------ #
class PrefixTransformer(nn.Module):
    def __init__(
        self,
        vocab_sz,
        embed_dim=128,
        nhead=2,
        layers=2,
        n_prefix=4,
        num_classes=2,
        pad_idx=0,
        max_len=512,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.n_prefix = n_prefix
        self.embed = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_len + n_prefix, embed_dim)
        self.prefix = nn.Parameter(torch.randn(n_prefix, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            embed_dim,
            nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x, pad_mask):
        bsz, seq_len = x.size()
        prefix = self.prefix.unsqueeze(0).repeat(bsz, 1, 1)  # (B, n_prefix, D)
        emb_seq = self.embed(x)
        emb = torch.cat([prefix, emb_seq], dim=1)  # (B, n_prefix+L, D)
        pos_ids = torch.arange(emb.size(1), device=x.device).unsqueeze(0)
        emb = emb + self.pos(pos_ids)
        # build new pad mask (False for prefix)
        new_mask = torch.cat(
            [
                torch.zeros((bsz, self.n_prefix), dtype=torch.bool, device=x.device),
                pad_mask,
            ],
            dim=1,
        )
        h = self.encoder(emb, src_key_padding_mask=new_mask)
        pooled = h[:, : self.n_prefix, :].mean(1)  # average prefix tokens
        return self.cls(pooled)


# -------------------- DATALOADERS ------------------------------- #
batch_size = 128
train_dl = DataLoader(
    dsets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate(b, vocab),
)
dev_dl = DataLoader(
    dsets["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)
test_dl = DataLoader(
    dsets["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)


# -------------------- TRAINING UTILS ---------------------------- #
def run_epoch(model, dl, criterion, optimizer=None):
    is_train = optimizer is not None
    total_loss, preds, gts = 0.0, [], []
    model.train() if is_train else model.eval()
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        if is_train:
            optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(gts)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# -------------------- TRAIN LOOP ------------------------------- #
model = PrefixTransformer(
    len(vocab),
    embed_dim=128,
    nhead=2,
    layers=2,
    n_prefix=4,
    num_classes=num_classes,
    pad_idx=pad_id,
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 8

for epoch in range(1, epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
    val_loss, val_f1, _, _ = run_epoch(model, dev_dl, criterion)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_F1 = {val_f1:.4f}  (train_F1={tr_f1:.4f})  time={time.time()-t0:.1f}s"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train"]["loss"].append(tr_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"]["f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"]["loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"]["f1"].append(val_f1)

# -------------------- TEST EVALUATION --------------------------- #
test_loss, test_f1, test_preds, test_gts = run_epoch(model, test_dl, criterion)
print(f"Test macro-F1 = {test_f1:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["test"] = {"loss": test_loss, "f1": test_f1}
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# -------------------- SAVE DATA --------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
