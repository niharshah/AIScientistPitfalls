import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from datasets import load_dataset, DatasetDict

# ----------------------------------------------------------- basic setup / paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------------- experiment container
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
        "SymToken_NoGate": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# ------------------------------------------------------------ dataset utilities
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    """Load full SPR benchmark csvs as HF datasets."""

    def _one(csv_file):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",  # read entire file as single split
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_one("train.csv"), dev=_one("dev.csv"), test=_one("test.csv")
    )


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
dsets = load_spr_bench(SPR_PATH)
print("Dataset sizes", {k: len(v) for k, v in dsets.items()})

# ------------------------------------------------------------ vocabulary, labels
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
    def __init__(self, hf_ds):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        ids, attn = encode_sequence(self.seq[idx])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(label2id[self.lab[idx]], dtype=torch.long),
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


# ------------------------------------------------------------ model definitions
class BaselineTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layer)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        x = self.emb(ids) + self.pos[:, : ids.size(1)]
        x = self.enc(x, src_key_padding_mask=~attn_mask.bool())
        return self.head(x[:, 0])


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
        sym_tok = self.sym_proj(counts).unsqueeze(1)
        cls_tok = self.token_emb(
            torch.full((B, 1), vocab[CLS], device=ids.device, dtype=torch.long)
        )
        tok_emb = self.token_emb(ids)
        x = torch.cat([cls_tok, sym_tok, tok_emb], dim=1) + self.pos[:, : L + 2]
        new_mask = torch.cat(
            [torch.ones(B, 2, device=ids.device, dtype=torch.long), attn_mask], dim=1
        )
        x = self.enc(x, src_key_padding_mask=~new_mask.bool())
        cls_out = x[:, 0] * self.gate(x[:, 0])
        return self.head(cls_out)


class SymbolicTokenTransformerNoGate(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.sym_proj = nn.Linear(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN + 2, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layer)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        B, L = ids.shape
        counts = torch.zeros(B, vocab_size, device=ids.device)
        ones = torch.ones_like(ids, dtype=torch.float)
        counts.scatter_add_(1, ids, ones)
        sym_tok = self.sym_proj(counts).unsqueeze(1)
        cls_tok = self.token_emb(
            torch.full((B, 1), vocab[CLS], device=ids.device, dtype=torch.long)
        )
        tok_emb = self.token_emb(ids)
        x = torch.cat([cls_tok, sym_tok, tok_emb], dim=1) + self.pos[:, : L + 2]
        new_mask = torch.cat(
            [torch.ones(B, 2, device=ids.device, dtype=torch.long), attn_mask], dim=1
        )
        x = self.enc(x, src_key_padding_mask=~new_mask.bool())
        return self.head(x[:, 0])


# ------------------------------------------------------------ training utilities
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optim=None):
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train_mode:
                optim.zero_grad()
                loss.backward()
                optim.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    macro_f1 = f1_score(gts, preds, average="macro")
    acc = accuracy_score(gts, preds)
    mcc = matthews_corrcoef(gts, preds)
    return tot_loss / len(loader.dataset), macro_f1, acc, mcc, preds, gts


def train(model, name, epochs=3, lr=3e-4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, tr_acc, tr_mcc, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_f1, val_acc, val_mcc, preds, gts = run_epoch(model, dev_loader)
        exp = experiment_data["SPR_BENCH"][name]
        exp["losses"]["train"].append({"epoch": ep, "loss": tr_loss})
        exp["losses"]["val"].append({"epoch": ep, "loss": val_loss})
        exp["metrics"]["train"].append(
            {"epoch": ep, "macro_f1": tr_f1, "MCC": tr_mcc, "ACC": tr_acc}
        )
        exp["metrics"]["val"].append(
            {"epoch": ep, "macro_f1": val_f1, "MCC": val_mcc, "ACC": val_acc}
        )
        print(
            f"{name} Ep{ep}: tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"trF1={tr_f1:.3f} valF1={val_f1:.3f} valMCC={val_mcc:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
    exp["predictions"], exp["ground_truth"] = preds, gts


# ------------------------------------------------------------ run all experiments
train(BaselineTransformer(), "Baseline")
train(SymbolicTokenTransformer(), "SymToken")
train(SymbolicTokenTransformerNoGate(), "SymToken_NoGate")

# ------------------------------------------------------------ persist results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
