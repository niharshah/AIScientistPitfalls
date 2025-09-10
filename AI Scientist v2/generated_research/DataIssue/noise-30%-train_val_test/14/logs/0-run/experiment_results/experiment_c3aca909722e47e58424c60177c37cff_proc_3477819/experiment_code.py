import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict

# ----------------------------- basic setup / utils
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ----------------------------- experiment data skeleton
def _empty_entry():
    return {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }


experiment_data = {
    "SPR_BENCH": {
        "Baseline": _empty_entry(),
        "SymToken": _empty_entry(),
        "MeanPool": _empty_entry(),
    }
}


# ----------------------------- dataset loading
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


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
dsets = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in dsets.items()})

# ----------------------------- vocabulary
CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}
for seq in dsets["train"]["sequence"]:
    for tok in seq.strip().split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)

labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
MAX_LEN = 128


# ----------------------------- encoders / dataset
def encode_sequence(seq: str, use_cls: bool = True):
    toks = seq.strip().split()
    ids = ([vocab[CLS]] if use_cls else []) + [vocab.get(t, vocab[UNK]) for t in toks]
    ids = ids[:MAX_LEN]
    attn = [1] * len(ids)
    pad = MAX_LEN - len(ids)
    if pad > 0:
        ids += [vocab[PAD]] * pad
        attn += [0] * pad
    return ids, attn


class SPRDataset(Dataset):
    def __init__(self, hf, use_cls: bool = True):
        self.seqs, self.labels = hf["sequence"], hf["label"]
        self.use_cls = use_cls

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        token_ids, attn = encode_sequence(self.seqs[idx], self.use_cls)
        return {
            "input_ids": torch.tensor(token_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(label2id[self.labels[idx]]),
        }


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


BATCH = 64
train_loader_cls = DataLoader(
    SPRDataset(dsets["train"], True), batch_size=BATCH, shuffle=True, collate_fn=collate
)
dev_loader_cls = DataLoader(
    SPRDataset(dsets["dev"], True), batch_size=BATCH, shuffle=False, collate_fn=collate
)
train_loader_mean = DataLoader(
    SPRDataset(dsets["train"], False),
    batch_size=BATCH,
    shuffle=True,
    collate_fn=collate,
)
dev_loader_mean = DataLoader(
    SPRDataset(dsets["dev"], False), batch_size=BATCH, shuffle=False, collate_fn=collate
)


# ----------------------------- models
class BaselineTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layer)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn):
        x = self.emb(ids) + self.pos[:, : ids.size(1)]
        x = self.enc(x, src_key_padding_mask=~attn.bool())
        return self.cls(x[:, 0])


class SymbolicTokenTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.sym_proj = nn.Linear(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN + 2, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layer)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn):
        B, L = ids.shape
        counts = torch.zeros(B, vocab_size, device=ids.device)
        ones = torch.ones_like(ids, dtype=torch.float)
        counts.scatter_add_(1, ids, ones)
        sym = self.sym_proj(counts).unsqueeze(1)
        cls_tok = self.tok_emb(torch.full((B, 1), vocab[CLS], device=ids.device))
        tok = self.tok_emb(ids)
        x = torch.cat([cls_tok, sym, tok], 1) + self.pos[:, : L + 2]
        new_mask = torch.cat([torch.ones(B, 2, device=ids.device), attn], 1)
        x = self.enc(x, src_key_padding_mask=~new_mask.bool())
        out = x[:, 0]
        gated = out * self.gate(out)
        return self.head(gated)


class MeanPoolTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layer)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn):
        x = self.emb(ids) + self.pos[:, : ids.size(1)]
        x = self.enc(x, src_key_padding_mask=~attn.bool())
        mask = attn.unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)


# ----------------------------- training helpers
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optim=None):
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    tot, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    macro_f1 = f1_score(gts, preds, average="macro")
    acc = accuracy_score(gts, preds)
    return tot / len(loader.dataset), macro_f1, acc, preds, gts


def train(model, name, train_loader, dev_loader, epochs=3, lr=3e-4):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        t = time.time()
        tr_loss, tr_f1, _, _, _ = run_epoch(model, train_loader, optim)
        val_loss, val_f1, val_acc, preds, gts = run_epoch(model, dev_loader)
        entry = experiment_data["SPR_BENCH"][name]
        entry["losses"]["train"].append({"epoch": ep, "loss": tr_loss})
        entry["losses"]["val"].append({"epoch": ep, "loss": val_loss})
        entry["metrics"]["train"].append({"epoch": ep, "macro_f1": tr_f1, "RGA": None})
        entry["metrics"]["val"].append(
            {"epoch": ep, "macro_f1": val_f1, "RGA": val_acc}
        )
        print(
            f"{name} Ep{ep}: trL={tr_loss:.4f} valL={val_loss:.4f} "
            f"trF1={tr_f1:.3f} valF1={val_f1:.3f} RGA={val_acc:.3f} "
            f"({time.time()-t:.1f}s)"
        )
    entry["predictions"], entry["ground_truth"] = preds, gts


# ----------------------------- run training
train(BaselineTransformer(), "Baseline", train_loader_cls, dev_loader_cls)
train(SymbolicTokenTransformer(), "SymToken", train_loader_cls, dev_loader_cls)
train(MeanPoolTransformer(), "MeanPool", train_loader_mean, dev_loader_mean)

# ----------------------------- save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
