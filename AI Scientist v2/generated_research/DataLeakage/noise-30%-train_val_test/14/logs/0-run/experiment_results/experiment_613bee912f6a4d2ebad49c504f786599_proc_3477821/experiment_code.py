import os, pathlib, random, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------ paths / utils
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -------------------------------------------------- experiment bookkeeping dict
experiment_data = {
    "SPR_BENCH": {
        "Baseline": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "SymToken": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "SymToken_Sinusoidal": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# ------------------------------------------------------------ dataset / encoding
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


def encode_sequence(seq: str):
    toks = seq.strip().split()
    ids = [vocab[CLS]] + [vocab.get(t, vocab[UNK]) for t in toks]
    ids = ids[:MAX_LEN]
    attn = [1] * len(ids)
    if len(ids) < MAX_LEN:
        pad = MAX_LEN - len(ids)
        ids += [vocab[PAD]] * pad
        attn += [0] * pad
    return ids, attn


class SPRDataset(Dataset):
    def __init__(self, hf):
        self.seqs, self.labels = hf["sequence"], hf["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        token_ids, attn = encode_sequence(self.seqs[idx])
        return {
            "input_ids": torch.tensor(token_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(label2id[self.labels[idx]]),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


BATCH = 64
train_loader = DataLoader(
    SPRDataset(dsets["train"]), batch_size=BATCH, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dsets["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate
)


# ----------------------------------------------------- sinusoidal positional enc
def build_sinusoidal_embeddings(length: int, dim: int, device=None):
    """Returns (1,length,dim) sinusoidal matrix."""
    position = torch.arange(length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # (1, length, dim)
    return pe if device is None else pe.to(device)


# ------------------------------------------------------------------- models
class BaselineTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layer)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        x = self.emb(ids) + self.pos[:, : ids.size(1)]
        x = self.enc(x, src_key_padding_mask=~attn_mask.bool())
        return self.cls(x[:, 0])


class SymbolicTokenTransformer(nn.Module):
    """Learnable positional embedding variant (original)."""

    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.sym_proj = nn.Linear(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN + 2, d_model))  # +2 CLS+sym
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layer)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        B, L = ids.shape
        counts = torch.zeros(B, vocab_size, device=ids.device)
        counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float))
        sym_tok = self.sym_proj(counts).unsqueeze(1)  # (B,1,d)
        cls_tok = self.token_emb(torch.full((B, 1), vocab[CLS], device=ids.device))
        tok_emb = self.token_emb(ids)
        x = torch.cat([cls_tok, sym_tok, tok_emb], dim=1) + self.pos[:, : L + 2]
        new_mask = torch.cat([torch.ones(B, 2, device=ids.device), attn_mask], dim=1)
        x = self.enc(x, src_key_padding_mask=~new_mask.bool())
        cls_out = x[:, 0]
        gated = cls_out * self.gate(cls_out)
        return self.head(gated)


class SymbolicTokenTransformerSinusoidal(nn.Module):
    """Sinusoidal positional embedding ablation: same as SymbolicTokenTransformer
    but replaces learnable pos with fixed sinusoid embeddings."""

    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.sym_proj = nn.Linear(vocab_size, d_model)
        # register fixed sinusoidal embeddings
        sin_pe = build_sinusoidal_embeddings(MAX_LEN + 2, d_model)
        self.register_buffer("pos", sin_pe, persistent=False)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layer)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        B, L = ids.shape
        counts = torch.zeros(B, vocab_size, device=ids.device)
        counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float))
        sym_tok = self.sym_proj(counts).unsqueeze(1)
        cls_tok = self.token_emb(torch.full((B, 1), vocab[CLS], device=ids.device))
        tok_emb = self.token_emb(ids)
        pe = self.pos[:, : L + 2]
        x = torch.cat([cls_tok, sym_tok, tok_emb], dim=1) + pe
        new_mask = torch.cat([torch.ones(B, 2, device=ids.device), attn_mask], dim=1)
        x = self.enc(x, src_key_padding_mask=~new_mask.bool())
        cls_out = x[:, 0]
        gated = cls_out * self.gate(cls_out)
        return self.head(gated)


# ---------------------------------------------------------------- training helpers
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    macro_f1 = f1_score(gts, preds, average="macro")
    acc = accuracy_score(gts, preds)
    return tot_loss / len(loader.dataset), macro_f1, acc, preds, gts


def train(model, name, epochs=3, lr=3e-4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_f1, val_acc, preds, gts = run_epoch(model, dev_loader)
        exp = experiment_data["SPR_BENCH"][name]
        exp["losses"]["train"].append({"epoch": ep, "loss": tr_loss})
        exp["losses"]["val"].append({"epoch": ep, "loss": val_loss})
        exp["metrics"]["train"].append({"epoch": ep, "macro_f1": tr_f1, "RGA": None})
        exp["metrics"]["val"].append({"epoch": ep, "macro_f1": val_f1, "RGA": val_acc})
        print(
            f"{name} Epoch {ep}: tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"trF1={tr_f1:.3f} valF1={val_f1:.3f} RGA={val_acc:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
    exp["predictions"], exp["ground_truth"] = preds, gts


# ------------------------------------------------------------------- run training
train(BaselineTransformer(), "Baseline")
train(SymbolicTokenTransformer(), "SymToken")
train(SymbolicTokenTransformerSinusoidal(), "SymToken_Sinusoidal")

# ------------------------------------------------------------ save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
