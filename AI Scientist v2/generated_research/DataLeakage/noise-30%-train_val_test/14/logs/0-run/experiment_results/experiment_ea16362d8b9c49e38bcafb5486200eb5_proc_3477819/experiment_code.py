import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------- paths / reproducibility
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------- experiment dict
def _blank_record():
    return {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }


experiment_data = {
    "SPR_BENCH": {
        "Baseline": _blank_record(),
        "SymToken": _blank_record(),
        "RelPosBias": _blank_record(),
    }
}


# ------------------------------------------------------------ dataset / encoding
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
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
        vocab.setdefault(tok, len(vocab))

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
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids, attn = encode_sequence(self.seqs[idx])
        return {
            "input_ids": torch.tensor(ids),
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


# -------------------------------------------------------------------- base models
class BaselineTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
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
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.sym_proj = nn.Linear(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN + 2, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layer)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        B, L = ids.shape
        counts = torch.zeros(B, vocab_size, device=ids.device)
        ones = torch.ones_like(ids, dtype=torch.float)
        counts.scatter_add_(1, ids, ones)
        sym = self.sym_proj(counts).unsqueeze(1)
        cls_tok = self.token_emb(torch.full((B, 1), vocab[CLS], device=ids.device))
        tok_emb = self.token_emb(ids)
        x = torch.cat([cls_tok, sym, tok_emb], 1) + self.pos[:, : L + 2]
        new_mask = torch.cat([torch.ones(B, 2, device=ids.device), attn_mask], 1)
        x = self.enc(x, src_key_padding_mask=~new_mask.bool())
        cls = x[:, 0] * self.gate(x[:, 0])
        return self.head(cls)


# -------------------------------------------------------- relative-position bias
class RelPosBias(nn.Module):
    """
    Learned bias for each head indexed by relative distance.
    """

    def __init__(self, num_heads, max_dist=MAX_LEN):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.bias = nn.Parameter(torch.zeros(num_heads, 2 * max_dist + 1))

    def _relative_positions(self, qlen, klen, device):
        ctx = torch.arange(qlen, device=device)[:, None]
        mem = torch.arange(klen, device=device)[None, :]
        rel = mem - ctx
        rel = torch.clamp(rel, -self.max_dist, self.max_dist) + self.max_dist
        return rel  # (qlen,klen)

    def forward(self, qlen, klen, device):
        idx = self._relative_positions(qlen, klen, device)  # (L,L)
        return self.bias[:, idx]  # (H,L,L)


class RelPosAttention(nn.Module):
    def __init__(self, d_model=128, n_head=4, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.bias = RelPosBias(n_head)
        self.scale = self.d_head**-0.5
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, L, _ = x.size()
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.n_head, self.d_head).transpose(1, 2)  # B,H,L,d
        k = k.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # B,H,L,L
        attn = attn + self.bias(L, L, x.device)  # add relative bias
        if key_padding_mask is not None:
            mask = (~key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)  # B,1,1,L
            attn = attn.masked_fill(mask, float("-inf"))
        attn_prob = self.drop(torch.softmax(attn, -1))
        out = torch.matmul(attn_prob, v)  # B,H,L,d
        out = out.transpose(1, 2).contiguous().view(B, L, self.n_head * self.d_head)
        return self.o_proj(out)


class RelPosEncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_head=4, ff=256, dropout=0.1):
        super().__init__()
        self.attn = RelPosAttention(d_model, n_head, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class RelativePosBiasTransformer(nn.Module):
    """
    Transformer that removes absolute embeddings and uses learned
    relative-distance bias in every self-attention layer.
    """

    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.layers = nn.ModuleList(
            [RelPosEncoderLayer(d_model, n_head, ff) for _ in range(n_layer)]
        )
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        x = self.emb(ids)  # no position vectors added here
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.cls(x[:, 0])


# ---------------------------------------------------------- training / evaluation
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optim=None):
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    macro_f1 = f1_score(gts, preds, average="macro")
    acc = accuracy_score(gts, preds)
    return total_loss / len(loader.dataset), macro_f1, acc, preds, gts


def train_model(model, name, epochs=3, lr=3e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _, _ = run_epoch(model, train_loader, opt)
        val_loss, val_f1, val_acc, preds, gts = run_epoch(model, dev_loader)
        rec = experiment_data["SPR_BENCH"][name]
        rec["losses"]["train"].append({"epoch": ep, "loss": tr_loss})
        rec["losses"]["val"].append({"epoch": ep, "loss": val_loss})
        rec["metrics"]["train"].append({"epoch": ep, "macro_f1": tr_f1, "RGA": None})
        rec["metrics"]["val"].append({"epoch": ep, "macro_f1": val_f1, "RGA": val_acc})
        print(
            f"{name} Ep{ep}: tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"trF1={tr_f1:.3f} valF1={val_f1:.3f} RGA={val_acc:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
    rec["predictions"], rec["ground_truth"] = preds, gts


# ----------------------------------------------------------- run all experiments
train_model(BaselineTransformer(), "Baseline")
train_model(SymbolicTokenTransformer(), "SymToken")
train_model(RelativePosBiasTransformer(), "RelPosBias")

# ------------------------------------------------------------- save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
